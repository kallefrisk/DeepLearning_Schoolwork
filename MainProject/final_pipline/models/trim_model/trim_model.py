import torch.nn as nn


class Recurrent_classifier(nn.Module):
    """
    LSTM/GRU model used to predict running frames for trimming.
    """
    def __init__(self, hidden_layers: list, layer_type="LSTM", dropout=0):
        super().__init__()

        input_size = 39
        rnn_class  = nn.LSTM if layer_type == "LSTM" else nn.GRU

        self.rnns  = nn.ModuleList()
        self.drops = nn.ModuleList()
        self.norms = nn.ModuleList()  # Add normalization layers

        sizes = [input_size] + hidden_layers
        for i in range(len(hidden_layers)):
            in_size = sizes[i] * 2 if i > 0 else sizes[i]  # *2 because bidirectional doubles output
            self.rnns.append(rnn_class(in_size, sizes[i + 1], batch_first=True, bidirectional=True))
            self.drops.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())

            # Add LayerNorm after each RNN (normalize over the hidden dimension)
            hidden_dim = sizes[i + 1] * 2  # *2 for bidirectional
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.fc_out = nn.Linear(hidden_layers[-1] * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        for rnn, drop, norm in zip(self.rnns, self.drops, self.norms):
            x, _ = rnn(x)
            x = norm(x)  # Apply layer normalization
            x = drop(x)

        return self.fc_out(x)  # (batch, seq_len, 1) — raw logits
