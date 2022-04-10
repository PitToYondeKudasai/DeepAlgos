import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ACModel(nn.Module):
    def __init__(self, input_dim, action_dim, mem_out_size=128, activation=nn.ReLU()):
        super(ACModel, self).__init__()    
        self.conv = Conv_unit(input_dim, [16, 32], 64, nn.ReLU()) 
        dim_out_conv = self.conv.get_conv_out_shape(input_dim)
        
        self.memory = nn.LSTM(dim_out_conv, mem_out_size)

        self.fc_pi = nn.Sequential(
            nn.Linear(mem_out_size,64),
            activation,
            nn.Linear(64, action_dim)
        )
        
        self.fc_v = nn.Sequential(
            nn.Linear(mem_out_size,64),
            activation,
            nn.Linear(64, 1)
        )
               
    def forward(self, *args):
        x = args[0]
        h_in, c_in = args[-2:]
        
        x = self.conv(x)
        out, (h_out, c_out) = self.memory(x, (h_in, c_in))
        
        v = self.fc_v(out)
        pi = self.fc_pi(out)
        pi = F.softmax(pi, dim=2).squeeze(1)
        
        return pi, v, h_out, c_out

class RND(nn.Module):
    def __init__(self, input_dim):
        super(RND, self).__init__() 
        self.ref_conv = Conv_unit(input_dim, [16, 32], 64, nn.ReLU())
        self.ref_conv.train(False)

        dim_out_conv = self.ref_conv.get_conv_out_shape(input_dim) 

        self.ref_net = nn.Sequential(
            nn.Linear(dim_out_conv, 512),
            nn.ReLU(), 
            nn.Linear(512, 218),
            nn.ReLU(),   
            nn.Linear(218,1)     
        )
        self.ref_net.train(False)

        self.pred_conv = Conv_unit(input_dim, [16, 32], 64, nn.ReLU())
        self.pred_net = nn.Sequential(
            nn.Linear(dim_out_conv, 256),
            nn.ReLU(), 
            nn.Linear(256, 1),
        )

    def forward(self, x):
        ref_x = self.ref_conv(x).squeeze(1)
        pred_x = self.pred_conv(x).squeeze(1)
        return self.ref_net(ref_x), self.pred_net(pred_x)

    def extra_reward(self, obs):
        r1, r2 = self.forward(obs)
        return (r1 - r2).abs().detach().numpy()[0][0]

    def loss(self, obs_t):
        r1_t, r2_t = self.forward(obs_t)
        return F.mse_loss(r2_t, r1_t).mean()
    
class Conv_unit(nn.Module):
    def __init__(self, input_dim, conv_dims, output_dim, activation=nn.ReLU()):
        super(Conv_unit, self).__init__()
        
        self.output_dim = output_dim
        
        conv_dims.insert(0, input_dim[0])
        conv_dims.append(output_dim)
        conv_module_list = []
        for i in range(len(conv_dims) - 1):
            conv_module_list.append(nn.Conv2d(conv_dims[i], conv_dims[i + 1], 2, 1))
            conv_module_list.append(activation)
        self.seq = nn.Sequential(*conv_module_list)
        
    def get_conv_out_shape(self, input_dim):
        x = torch.zeros([1, *input_dim])
        x = self.seq(x)
        return np.prod(x.shape)
    
    def forward(self, x):
        x = self.seq(x).view(x.size()[0], 1, -1)
        return x
