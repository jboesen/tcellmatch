import torch.nn as nn

class biGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x