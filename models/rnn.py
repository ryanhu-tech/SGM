import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models


class rnn_encoder(nn.Module):

    def __init__(self, config, embedding=None):
        super(rnn_encoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)

    def forward(self, inputs, lengths):
        #對不等長的句子要進行pack後，會將pad=0的資料不納入運算
        embs = pack(self.embedding(inputs), lengths)
        outputs, state = self.rnn(embs)
        #將資料再轉回原本的排列方式
        outputs = unpack(outputs)[0]

        #為何要自己處理?與LSTM設定後取出的差在哪?
        #因為沒有要分別使用正反兩個結果，這邊是將正反的結果再合成一個結果
        #維度會跟只算單向相同
        if self.config.bidirectional:
            # 原本是[max_src_len, batch_size, 2*hidden_size]，把正反向的結果加總結合
            # outputs: [max_src_len, batch_size, hidden_size]
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            if self.config.cell == 'gru':
                state = state[:self.config.dec_num_layers]
            else:
                #state[0]=h_n, state[1]=c_n, state=[2*num_layers, batch, hidden_size]
                #[::2]是以step為2取資料，只取出正向序列的資料，但為何是取到第0個維度，雖然這是想要的結果
                state = (state[0][::2], state[1][::2])

        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None, use_attention=True):
        super(rnn_decoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.emb_size)
        #可以利用這樣的方法變動輸入的emb_size
        input_size = 2 * config.emb_size if config.global_emb else config.emb_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)

        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, input_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size, input_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, input_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.global_emb:
            self.ge_proj1 = nn.Linear(config.emb_size, config.emb_size)
            self.ge_proj2 = nn.Linear(config.emb_size, config.emb_size)
            self.softmax = nn.Softmax(dim=1)

    #input[batch], state[2][num_layers=3, batch, hidden_size]
    def forward(self, input, state, output=None, mask=None): 
        #embs=[batch, emb_size]
        embs = self.embedding(input)

        if self.config.global_emb:
            if output is None:
                output = embs.new_zeros(embs.size(0), self.config.tgt_vocab_size)#為何tgt_vocab_size是107
            # 為何要將output除以0.1為分母做softmax，這樣不是等於放大10倍的意思
            # paper中沒寫到這個操作
            probs = self.softmax(output / self.config.tau)
            #公式(11)
            emb_avg = torch.matmul(probs, self.embedding.weight)
            #公式(13)
            H = torch.sigmoid(self.ge_proj1(embs) + self.ge_proj2(emb_avg))
            #公式(12)
            emb_glb = H * embs + (1 - H) * emb_avg
            #-1表示把最後一個維度串接起來[tgt_len, 2*emb_size]
            embs = torch.cat((embs, emb_glb), dim=-1)
        #output=[batch, hidden_size]
        #state=([3,batch, hidden_size],[3,batch, hidden_size])
        #此部分應該是公式(7)，但是僅用到s_t-1及g(y_t-1)，
        #可能是在算g(y_t-1)已經用到c_t-1的資訊(output)
        #而output又是經過mask label的計算
        output, state = self.rnn(embs, state)
        #此部分是公式(6)出來的結果
        if self.attention is not None:
            if self.config.attention == 'luong_gate':
                #attn_weights=[batch, max_src_length]
                output, attn_weights = self.attention(output)
            else:
                output, attn_weights = self.attention(output, embs)
        else:
            attn_weights = None
        #把算完的y_t又轉換一次，是公式(8)
        #output=[batch,hidden_size]->[batch,tgt_vocab_size]
        output = self.compute_score(output)

        #如papaer中說的，將預測出來的標籤值mask掉，用很小的值取代掉
        #公式(9)
        if self.config.mask and mask:
            mask = torch.stack(mask, dim=1).long()
            output.scatter_(dim=1, index=mask, value=-1e7)
        #output = [batch, tgt_vocab_size],state=([3,batch, hidden_size],[3,batch, hidden_size])
        return output, state, attn_weights

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores

#連需堆疊指定數量的lstm，利用LSTMCell創一個不同的LSTM
class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        #這裡的num_layers指的是重複幾個單一layer的lstm
        for _ in range(num_layers):
            lstm = nn.LSTMCell(input_size, hidden_size)
            self.layers.append(lstm)
            input_size = hidden_size

    def forward(self, input, hidden):
        # input=[batch, 2*emb_size]
        # hidden=[num_layer=3, batch, hidden_size]
        # h_0[0], h_0[1] 跟layer[0],layer[1]的關係?，為何可做為每個layer的input
        # 每一層對應的是encoder的layer，paper沒解釋為何要這樣對應
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            #input=[batch, 2*emb_size],h_1_i=[batch, hidden_size]
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            #input =[batch, hidden_size]
            input = h_1_i
            if i + 1 != self.num_layers:#若不是最後一層，都要dropout
                input = self.dropout(input)
            h_1 += [h_1_i] #把當前這層的結果與上一層相加(是表示串接下去)
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1) #直接疊加[[h_1_1], [h_1_2],...] = [3,batch, hidden_size]
        c_1 = torch.stack(c_1)
        #input=[batch, hidden_size], ([3,batch, hidden_size],[3,batch, hidden_size])
        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1
