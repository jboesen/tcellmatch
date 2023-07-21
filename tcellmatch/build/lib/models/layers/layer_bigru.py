import torch.nn as nn

class biGRU(nn.Module):
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        hidden_dim: int, dropout: int = 0,
        return_sequences: bool = True
    ):
        super(BiGRU, self).__init__()
        self.return_sequences = return_sequences
        self.gru = nn.GRU(
            input_dim,
            w,
            num_layers=1,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, x):
        x, _ = self.gru(x)
        if not self.return_sequences:
            x = x[:, -1, :]  # Select only the output for the last time step
        return x