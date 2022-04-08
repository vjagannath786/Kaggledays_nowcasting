import torch.nn as nn

from sklearn.metrics import mean_squared_error

def loss_fn():
    pass


class NoCModel(nn.Module):
    def __init__(self):
        super(NoCModel, self).__init__()
        self.lstm_1 = nn.LSTM(45,128)
        self.linear = nn.Linear(128,1)
        self.relu = nn.ReLU(128)
    
    def forward(self, x, targets=None):

        x,_ = self.lstm_1(x)

        x = self.linear(x)

        x = self.relu(x)

        if targets is None:
            return x
        else:
            loss = loss_fn(targets, x)
            return x, loss

