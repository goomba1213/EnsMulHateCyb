import torch.nn as nn
import torch.nn.functional as F
import torch

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, **kwargs):
        super(BiLSTM, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0.3)

        if self.bidirectional:
            self.linear1 = nn.Linear(num_hiddens * 4, labels)
        else:
            self.linear1 = nn.Linear(num_hiddens * 2, labels)



    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        #print(embeddings.shape)
        #states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        output, (h_n, c_n) = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([output[0], output[-1]], dim=1) #if it's bidirectional, choose first and last output
        res = self.linear1(encoding)

        return res
    
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, **kwargs):
        super(BiGRU, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.GRU(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0.3)

        if self.bidirectional:
            self.linear1 = nn.Linear(num_hiddens * 4, labels)
        else:
            self.linear1 = nn.Linear(num_hiddens * 2, labels)



    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        #print(embeddings.shape)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1) #if it's bidirectional, choose first and last output
        outputs = self.linear1(encoding)

        return outputs

class CNNLSTM(nn.Module):
    #def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, **kwargs):
    def __init__(self, vocab_size, embed_size, seq_len, labels, weight, num_hiddens, num_layers, bidirectional, **kwargs):
        super(CNNLSTM, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.labels = labels
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        #in (1,x,emb)
        self.conv1 = nn.Conv2d(1, 10, (3, embed_size))
        #out (10,x-2,emb)

        self.dropout = nn.Dropout(p=0.5)

        self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))

        self.lstm = nn.LSTM(input_size=10, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0.3)

        if self.bidirectional:
            self.linear1 = nn.Linear(num_hiddens * 4, labels)
        else:
            self.linear1 = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        inputs = self.embedding(inputs).view(inputs.shape[0], 1, inputs.shape[1], -1)
        x1 = F.relu(self.conv1(inputs))
        x1 = self.pool1(x1)

        x = x1.squeeze(dim=-1)
        states, hidden = self.lstm(x.permute([2, 0, 1]))
        temp = []
        encoding = torch.cat([states[0], states[-1]], dim=1) #if it's bidirectional, choose first and last output

        #outputs = self.linear1(cated)
        outputs = self.linear1(encoding)
        #x = self.linear(x)
        #x = x.view(-1, self.labels)

        return(outputs)