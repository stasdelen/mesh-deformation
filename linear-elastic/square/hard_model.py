import torch
from torch import nn

class HardMesh(nn.Module):
    def __init__(self, distance, Up):
        super().__init__()
        # Distance function
        self.distance = distance
        self.Xp, self.Yp = Up[:,0:1], Up[:,1:2]
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            # nn.Linear(50, 50),
            # nn.Tanh(),
            # nn.Linear(50, 50),
            # nn.Tanh(),
            # nn.Linear(50, 50),
            # nn.Tanh(),
            # nn.Linear(50, 50),
            # nn.Tanh(),
            # nn.Linear(50, 50),
            # nn.Tanh(),
            # nn.Linear(50, 50),
            # nn.Tanh(),
            nn.Linear(50, 2)
        )
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)
        self.linear_sigmoid_stack.apply(init_weights)
    
    def forward(self, x, y):
        Upar = torch.hstack((x, y))
        Uhat = self.linear_sigmoid_stack(Upar)
        X_nn, Y_nn = Uhat[:,0:1], Uhat[:,1:2]
        X = self.Xp + self.distance * X_nn
        Y = self.Yp + self.distance * Y_nn
        # R(X,Y) -> dR/d0 = dR/dX * dX/d0 + ...
        return X, Y
    
    def calc(self, Uin):
        Uhat = self.linear_sigmoid_stack(Uin)
        X_nn, Y_nn = Uhat[:,0:1], Uhat[:,1:2]
        return torch.hstack(
            (self.Xp + self.distance * X_nn,
             self.Yp + self.distance * Y_nn)
        )