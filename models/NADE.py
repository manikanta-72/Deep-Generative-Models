import torch
from torch import nn


class NADE(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        """
            out_dim = in_dim
        """
        super(NADE, self).__init__()
        self.input_dim = input_dim

        self.w = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim,1))
        self.v = nn.Parameter(torch.zeros(input_dim, hidden_dim))
        self.c = nn.Parameter(torch.zeros(hidden_dim,1))

    def forward(self, x):
        '''
            generate new image
        '''
        a = self.c
        p_x = []
        x_s = []
        for i in range(self.input_dim):
            h_i = torch.sigmoid(a)
            # print("v: ",self.v[i:i+1,:].shape)
            # print("h_i: ",h_i.shape)
            p_i = torch.sigmoid(torch.mm(self.v[i:i+1,:],h_i) + self.b[i:i+1])
            if x is not None:
                x_i = x[i:i+1]
            else:
                # sample x based on p_i
                x_i = torch.distributions.Bernoulli(probs = p_i).sample()
                xs.append(x_i)
            p_x.append(p_i)
            # print("w: ",self.w[:,i:i+1].shape)
            # print("x_i: ",x_i.shape)
            # print("x_i: ", x_i)
            # print("i: ", i)
            a = torch.mm(self.w[:,i:i+1],x_i) + a

        p_x = torch.cat(p_x, 0)
        if(x is None):
            x_s = torch.cat(x_s, 0)
        return p_x, x_s
