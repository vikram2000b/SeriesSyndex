from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout = dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        packed_output,(hidden_state,cell_state) = self.lstm(x)
        return self.sigmoid(self.fc(hidden_state[-1]))
    

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout = dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        packed_output,(hidden_state,cell_state) = self.lstm(x)
        return self.fc(hidden_state[-1])

