import torch
import torch.nn as nn


# 1. SIMPLE RNN MODEL
class SimpleRNNModel(nn.Module):
    """
    A simple Elman RNN for sequence classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, nonlinearity='tanh')

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_size)
        Returns:
            out shape:
        """

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, hn = self.rnn(x, h0)

        last_out = out[:, -1, :]  # (batch, hidden_size)

        output = self.fc(last_out)
        return output
