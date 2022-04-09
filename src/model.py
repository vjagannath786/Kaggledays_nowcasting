import torch.nn as nn


from sklearn.metrics import mean_squared_error

def loss_fn(x, targets):
    #print(targets.unsqueeze(1).shape)
    #print(x.shape)
    loss = nn.MSELoss()
    return loss(x,targets.unsqueeze(1))


class NoCModel(nn.Module):
    def __init__(self):
        super(NoCModel, self).__init__()
        self.lstm_1 = nn.LSTM(16,128, num_layers=2, batch_first=True)
        self.linear = nn.Linear(128,1)
        self.relu = nn.ReLU(128)
    
    def forward(self, x, targets=None):

        

        

        x,_ = self.lstm_1(x)

        x = self.linear(x[:,-1,:])

        #x = self.relu(x[:,-1,:])

        if targets is None:
            #print('Im here')
            return x
        else:
            loss = loss_fn(x,targets)
            return x, loss

