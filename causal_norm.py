import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class CumulativeLayerNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1,1,1,num_features))
            self.bias = nn.Parameter(torch.zeros(1,1,1,num_features))
        else:
            self.weight = Variable(torch.ones(1,1,1,num_features), requires_grad=False)
            self.bias = Variable(torch.zeros(1,1,1,num_features), requires_grad=False)
    
    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1,3], keepdim=True)  # (B,1,T,1)
        step_pow_sum = inpt.pow(2).sum([1,3], keepdim=True)  # (B,1,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,1,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,1,T,1)

        # entry_cnt = np.arange(channel*freq_num, channel*freq_num*(seq_len+1), channel*freq_num)
        # entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = torch.arange(channel*freq_num, channel*freq_num*(seq_len+1), channel*freq_num, 
                                 dtype=torch.float, device=inpt.device).type(inpt.type()) # (T, ): [B, 2*B, ..., T*B]
        entry_cnt = entry_cnt.view(1,1,seq_len,1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.weight.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features), requires_grad=True)
        else:
            self.weight = Variable(torch.ones(1, 1, num_features), requires_grad=False)
            self.bias = Variable(torch.zeros(1, 1, num_features), requires_gra=False)


    def forward(self, input):
        """
        Args:
            input (B, T, C):
        Returns:
            output (B, T, C): same shape as the input
        """
        n_dims = input.dim()
        if n_dims == 3:
            B, T, C = input.size()
        else:
            raise ValueError("Only support 3D input, but given {}D".format(input.dim()))


        step_sum = torch.sum(input, dim=2) # (batch_size, T)
        step_squared_sum = torch.sum(input**2, dim=2) # (batch_size, T)
        cum_sum = torch.cumsum(step_sum, dim=1) # (batch_size, T)
        cum_squared_sum = torch.cumsum(step_squared_sum, dim=1) # (batch_size, T)

        entry_cnt = torch.arange(C, C * (T + 1), C, 
                                 dtype=torch.float, device=input.device).type(input.type()) # (T, ): [C, 2*C, ..., T*C]
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # (batch_size, T)
        cum_var = (cum_squared_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # (batch_size, T)
        cum_std = (cum_var + self.eps).sqrt()  # (batch_size, T)

        cum_mean, cum_std = cum_mean.unsqueeze(dim=2), cum_std.unsqueeze(dim=2) # (batch_size, T, 1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        output = x * self.weight.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())

        return output
    

class CumulativeInstanceNorm2d(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super(CumulativeInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
            self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        else:
            self.weight = Variable(torch.ones(1,num_features,1,1), requires_grad=False)
            self.bias = Variable(torch.zeros(1,num_features,1,1), requires_grad=False)

    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([3], keepdim=True)  # (B,C,T,1)
        step_pow_sum = inpt.pow(2).sum([3], keepdim=True)  # (B,C,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,C,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,C,T,1)

        entry_cnt = np.arange(freq_num, freq_num*(seq_len+1), freq_num)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1,1,seq_len,1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.weight.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeBatchNorm1d(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super(CumulativeBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.weight = nn.Parameter(torch.ones(1,num_features,1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1,num_features,1), requires_grad=True)
        else:
            self.weight = Variable(torch.ones(1,num_features,1), requires_grad=False)
            self.bias = Variable(torch.zeros(1,num_features,1), requires_gra=False)

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape

        step_sum = torch.sum(inpt, dim=0) # (C, T)
        step_squared_sum = torch.sum(inpt**2, dim=0) # (C, T)
        cum_sum = torch.cumsum(step_sum, dim=1) # (C, T)
        cum_squared_sum = torch.cumsum(step_squared_sum, dim=1) # (C, T)

        entry_cnt = torch.arange(b_size, b_size * (seq_len + 1), b_size, 
                                 dtype=torch.float, device=inpt.device).type(inpt.type()) # (T, ): [B, 2*B, ..., T*B]
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # (C, T)
        cum_var = (cum_squared_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # (C, T)
        cum_std = (cum_var + self.eps).sqrt()  # (C, T)

        cum_mean, cum_std = cum_mean.unsqueeze(dim=0), cum_std.unsqueeze(dim=0) # (1, C, T)
        
        x = (inpt - cum_mean) / cum_std
    
        output = x * self.weight.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
        # output = x *self.weight.type(x.type()) + self.bias.type(x.type())
        return output


class CumulativeBatchNorm2d(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super(CumulativeBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.weight = nn.Parameter(torch.ones(1,num_features,1,1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1,num_features,1,1), requires_grad=True)
        else:
            self.weight = Variable(torch.ones(1,num_features,1,1), requires_grad=False)
            self.bias = Variable(torch.zeros(1,num_features,1,1), requires_gra=False)

    def forward(self, inpt):
        # inpt: (B,C,T,F)
        b_size, channel, seq_len, freq = inpt.shape

        step_sum = torch.sum(inpt, dim=[0,3]) # (C, T)
        step_squared_sum = torch.sum(inpt**2, dim=[0,3]) # (C, T)
        cum_sum = torch.cumsum(step_sum, dim=1) # (C, T)
        cum_squared_sum = torch.cumsum(step_squared_sum, dim=1) # (C, T)

        entry_cnt = torch.arange(b_size*freq, b_size*freq * (seq_len + 1), b_size*freq, 
                                 dtype=torch.float, device=inpt.device).type(inpt.type()) # (T, ): [B*F, 2*B*F, ..., T*B*F]
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # (C, T)
        cum_var = (cum_squared_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # (C, T)
        cum_std = (cum_var + self.eps).sqrt()  # (C, T)

        cum_mean, cum_std = cum_mean.unsqueeze(0).unsqueeze(3), cum_std.unsqueeze(0).unsqueeze(3) # (1, C, T, 1)
        
        x = (inpt - cum_mean) / cum_std
    
        output = x * self.weight.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
        return output
    

if __name__  == "__main__":
    data = torch.rand(4, 100, 80)
    net = CumulativeLayerNorm1d(80)
    out = net(data)
    print(data.shape)
    print(out.shape)


    
    data = torch.rand(4, 2, 100, 80)
    net = CumulativeLayerNorm2d(80)
    out = net(data)
    print(data.shape)
    print(out.shape)


    data = torch.rand(4, 80, 100)
    net = CumulativeBatchNorm1d(80)
    out = net(data)
    print(data.shape)
    print(out.shape)


    data = torch.rand(4, 2, 100, 80)
    net = CumulativeBatchNorm2d(2)
    out = net(data)
    print(data.shape)
    print(out.shape)