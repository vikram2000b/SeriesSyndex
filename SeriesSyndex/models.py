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
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout = dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        packed_output,(hidden_state,cell_state) = self.lstm(x)
        return self.fc(hidden_state[-1])
    
class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, num_layers, dropout = 0.5, dilation_factor=2):
        '''
        Contruct the Temporal CNN Classifier.
        Args:
            input_size (int): The dimension of the input.
            num_channels (int): Number of channels for the CNNs.
            kernel_size (int): Kernel Size for CNNs.
            num_layer (int): Number of layers in the network
        '''

        super(TCNClassifier, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = dilation_factor ** i
            layers.extend(
                [
                    nn.Conv1d(input_size, num_channels, kernel_size, padding = (kernel_size-1)*dilation_size, dilation=dilation_size),#, padding=(kernel_size - 1) // 2),
                    nn.ELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_size = num_channels
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))
        x = x.mean(dim=2)  # Global average pooling
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class TCNRegressor(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, num_layers, dropout = 0.5, dilation_factor=2):
        '''
        Contruct the Temporal CNN Classifier.
        Args:
            input_size (int): The dimension of the input.
            num_channels (int): Number of channels for the CNNs.
            kernel_size (int): Kernel Size for CNNs.
            num_layer (int): Number of layers in the network
        '''

        super(TCNRegressor, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = dilation_factor ** i
            layers.extend(
                [
                    nn.Conv1d(input_size, num_channels, kernel_size, padding = (kernel_size-1)*dilation_size, dilation=dilation_size), #padding=(kernel_size - 1) // 2,
                    nn.ELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_size = num_channels
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))
        x = x.mean(dim=2)  # Global average pooling
        x = self.fc(x)
        return x

