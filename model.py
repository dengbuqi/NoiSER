import torch
import torch.nn as nn

class NoiSER(nn.Module):
    def __init__(self, hidden_channel=32):
        super(NoiSER,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3,hidden_channel,3,1,1),
            nn.InstanceNorm2d(hidden_channel),
            nn.GELU(),
            nn.Conv2d(hidden_channel,3,3,1,1),
            nn.Tanh(),
        )
    def forward(self,x):
        return self.network(x)