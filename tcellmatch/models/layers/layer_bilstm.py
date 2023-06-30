import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        hidden_dim: int,
        dropout: int = 0,
        return_sequences: bool = True,
        n_layers: int = 1
    ):
        super(BiLSTM, self).__init__()
        self.return_sequences = return_sequences
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        if not self.return_sequences:
            x = x[:, -1, :]  # Select only the output for the last time step
        return x