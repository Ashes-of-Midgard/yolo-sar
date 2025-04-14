""" DCT Deno Attention
    from: https://arxiv.org/abs/2406.02833
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectBlock(nn.Conv1d):
    def __init__(self, channels, branches):
        super(SelectBlock, self).__init__(channels,branches,1, bias=False)
        self.eps = 1e-15
        self.branches = branches
        self.softmax = nn.Softmax(1)

    def forward(self, origin_tensor, branch_tensors):
        branch_offsets = super().forward(origin_tensor)
        branch_offsets = branch_offsets - branch_offsets.min(1,keepdim=True)[0] + self.eps
        branch_offsets = branch_offsets / branch_offsets.max(1,keepdim=True)[0] * (branch_tensors.size(1) - 1)

        b,c,h,w = branch_tensors.size()
        y = branch_tensors.clone()
        branch_min = (branch_offsets.floor().long()).view(b, self.branches, 1, 1).expand(b, self.branches, h, w)
        branch_max = branch_offsets.ceil().long().view(b, self.branches, 1, 1).expand(b, self.branches, h, w)
        min_offset = (branch_offsets - branch_offsets.floor()).view(b, self.branches, 1, 1).expand(b,self.branches, h, w)
        max_offset = (branch_offsets.ceil() - branch_offsets).view(b, self.branches, 1, 1).expand(b,self.branches, h, w)
        offset = self.softmax(torch.cat([min_offset,max_offset],dim=1)).split(self.branches,dim=1)
        min_offset = offset[0]
        max_offset = offset[1]
        for i in range(self.branches):
            y[:,i,...] = (torch.gather(branch_tensors, 1, branch_min[:,i,...].unsqueeze(1)).squeeze(1) * min_offset[:,i,...]
                          + torch.gather(branch_tensors, 1, branch_max[:,i,...].unsqueeze(1)).squeeze(1) * max_offset[:,i,...])

        return y.sum(1)


class SelectGroupFClayer(nn.Module):
    def __init__(self, channel):
        super(SelectGroupFClayer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, groups=2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, groups=4),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, groups=8),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, groups=16),
            nn.ReLU(),
        )
        self.select1 = SelectBlock(channel,4)
        
        self.fc5 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, groups=2),
            nn.Sigmoid(),
        )
        self.fc6 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, groups=4),
            nn.Sigmoid(),
        )
        self.fc7 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, groups=8),
            nn.Sigmoid(),
        )
        self.fc8 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, groups=16),
            nn.Sigmoid(),
        )
        self.select2 = SelectBlock(channel,4)
    
    def forward(self, x):
        b,c,h,w = x.size()
        y = x.clone().reshape(b,h*w,c)
        y = y.mean(2, keepdim=True) + y.max(2, keepdim=True)[0]

        y1 = self.fc1(y).unsqueeze(1)
        y2 = self.fc2(y).unsqueeze(1)
        y3 = self.fc3(y).unsqueeze(1)
        y4 = self.fc4(y).unsqueeze(1)
        temp = self.select1(y, torch.cat([y1,y2,y3,y4],dim=1)) 
        
        y5 = self.fc5(temp).unsqueeze(1)
        y6 = self.fc6(temp).unsqueeze(1)
        y7 = self.fc7(temp).unsqueeze(1)
        y8 = self.fc8(temp).unsqueeze(1)
        att = self.select2(temp, torch.cat([y5,y6,y7,y8],dim=1)) 
        
        return att.view(b,1,h,w).expand_as(x) * x


class SimpleFCLayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.fc = nn.Conv1d(channel, channel, 1)
    
    def forward(self, x):
        b,c,h,w = x.size()
        x = x.permute(0,2,3,1).reshape(b,h*w,c)
        x = self.fc(x)
        x = x.reshape(b,h,w,c).permute(0,3,1,2)
        return x
    

class AdaFreqMaskLayer(nn.Module):
    def __init__(self, in_channels, width, height):
        super().__init__()
        self.mask = nn.Parameter(torch.ones([in_channels, width, height]))
    
    def forward(self, x):
        return x * self.mask.unsqueeze(0)


class SqueezeFCLayer(nn.Module):
    def __init__(self, channel, latent_channel):
        super().__init__()
        self.fc1 = nn.Conv1d(channel, latent_channel, 1)
        self.fc2 = nn.Conv1d(latent_channel, channel, 1)
    
    def forward(self, x):
        b,c,h,w = x.size()
        x = x.permute(0,2,3,1).reshape(b,h*w,c)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape(b,h,w,c).permute(0,3,1,2)
        return x


def get_dct_weight(length, device="cpu"):
    k = torch.arange(length, device=device)
    n = k.unsqueeze(-1)
    dct_weight = torch.cos(torch.pi * (0.5 + n) * k / length) / torch.sqrt(torch.tensor(length)).to(device)
    dct_weight[:,1:] *= torch.sqrt(torch.tensor(2)).to(device)
    return dct_weight


class DCT2d(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.dct_matrix_x = nn.Parameter(get_dct_weight(width), requires_grad=False)
        self.dct_matrix_y = nn.Parameter(get_dct_weight(height), requires_grad=False)
    
    def forward(self, x):
        X = torch.matmul(x, self.dct_matrix_x)
        X = torch.matmul(self.dct_matrix_y.transpose(-1, -2), X)
        return X
    

class IDCT2d(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.dct_matrix_x = nn.Parameter(get_dct_weight(width), requires_grad=False)
        self.dct_matrix_y = nn.Parameter(get_dct_weight(height), requires_grad=False)
    
    def forward(self, X):
        X = torch.matmul(X, self.dct_matrix_x.transpose(-1, -2))
        x = torch.matmul(self.dct_matrix_y, X)
        return x


class DCTDenoAttention(nn.Module):
    def __init__(self, in_channels, width, height):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.height = height
        self.dct = DCT2d(width, height)
        # self.fc = SelectGroupFClayer(width*height)
        # self.fc = SimpleFCLayer(width*height) # a simpler version of fc layer
        # self.fc = AdaFreqMaskLayer(in_channels, width, height) # an adaptive frequency mask
        self.fc = SqueezeFCLayer(width*height, 32)
        self.idct = IDCT2d(width, height)
        self.drop_out = nn.Dropout2d()
        self.norm = nn.LayerNorm([in_channels, height, width])

    def forward(self, x):
        ori_shape = None
        if x.shape[-1] != self.height or x.shape[-2] != self.width:
            ori_shape = x.shape
            x = F.interpolate(x, [self.width, self.height])
        X = self.dct(x)
        Y = self.fc(X)
        y = self.idct(Y)
        # x = x + self.drop_out(y)
        # x = self.norm(x)
        x = torch.concat([x, y], dim=1)
        if ori_shape:
            x = F.interpolate(x, [ori_shape[-1], ori_shape[-2]])
        return x


if __name__=="__main__":
    # test DCT
    import numpy as np
    from scipy.fft import dctn, idctn

    dct2d  = DCT2d(32, 16).cpu().eval()
    idct2d = IDCT2d(32, 16).cpu().eval()

    x_torch = torch.randn(1, 16, 32)
    x_np = x_torch.numpy()

    X_scipy = dctn(x_np, type=2, norm='ortho')
    X_torch = dct2d(x_torch).numpy()

    print('DCT max abs diff:', np.max(np.abs(X_scipy - X_torch)))

    x_rec_scipy = idctn(X_scipy, type=2, norm='ortho')
    x_rec_torch = idct2d(torch.from_numpy(X_scipy)).numpy()

    print('IDCT max abs diff:', np.max(np.abs(x_rec_scipy - x_rec_torch)))
