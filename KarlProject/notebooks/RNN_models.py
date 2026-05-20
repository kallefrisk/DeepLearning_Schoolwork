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
            out shape: (batch, num_classes)
        """

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, hn = self.rnn(x, h0)

        last_out = out[:, -1, :]  # (batch, hidden_size)

        output = self.fc(last_out)
        return output

    def to_string(self):
        return "SimpleRNNModel"


# 2. LSTM MODEL
class LSTMModel(nn.Module):
    """
    LSTM model for sequence classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_size)
        Returns:
            out_shape: (batch, num_classes)
        """

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        last_out = out[:, -1, :]  # (batch, hidden_size)

        output = self.fc(last_out)
        return output

    def to_string(self):
        return "LSTMModel"


# 3. GRU MODEL
class GRUModel(nn.Module):
    """
    GRU model for sequence classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.0):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_size)
        Returns:
            out_shape: (batch, num_classes)
        """

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)

        last_out = out[:, -1, :]  # (batch, hidden_size)

        output = self.fc(last_out)
        return output

    def to_string(self):
        return "GRUModel"
