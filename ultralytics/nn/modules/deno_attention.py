""" DCT Deno Attention
    from: https://arxiv.org/abs/2406.02833
"""
# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCT2DSpatialTransformLayer(nn.Module):
    def __init__(self, width, height):
        super(DCT2DSpatialTransformLayer, self).__init__()
        self.dct_x = DCT2DSpatialTransformLayer_x(width)
        self.dct_y = DCT2DSpatialTransformLayer_y(height)

    def forward(self, x):
        # x = x.double()
        y = self.dct_x(x)
        y = self.dct_y(y)

        return y


class IDCT2DSpatialTransformLayer(nn.Module):
    def __init__(self, width, height):
        super(IDCT2DSpatialTransformLayer, self).__init__()
        self.idct_x = IDCT2DSpatialTransformLayer_x(width)
        self.idct_y = IDCT2DSpatialTransformLayer_y(height)

    def forward(self, x):
        y = self.idct_x(x)
        y = self.idct_y(y)
        # y = y.float()

        return y


class FastDCT2DSpatialTransformLayer(nn.Module):
    def __init__(self, width, height):
        super(FastDCT2DSpatialTransformLayer, self).__init__()
        self.dct_x = FastDCT2DSpatialTransformLayer_x(width)
        self.dct_y = FastDCT2DSpatialTransformLayer_y(height)

    def forward(self, x):
        # x = x.double()
        # b,c,h,w
        y = self.dct_x(x.permute(0, 3, 2, 1))  # channel 和 width 互换 b,w,h,c
        y = self.dct_y(y.permute(0, 2, 1, 3))  # width 和 height 互换 b,h,w,c
        y = y.permute(0, 3, 1, 2)  # 复原 b,c,h,w

        return y


class FastIDCT2DSpatialTransformLayer(nn.Module):
    def __init__(self, width, height):
        super(FastIDCT2DSpatialTransformLayer, self).__init__()
        self.idct_x = FastIDCT2DSpatialTransformLayer_x(width)
        self.idct_y = FastIDCT2DSpatialTransformLayer_y(height)

    def forward(self, x):
        # b,c,h,w
        y = self.idct_x(x.permute(0, 3, 2, 1))  # channel 和 width 互换 b,w,h,c
        y = self.idct_y(y.permute(0, 2, 1, 3))  # width 和 height 互换 b,h,w,c
        y = y.permute(0, 3, 1, 2)  # 复原 b,c,h,w
        # y = y.float()

        return y


class DCT2DSpatialTransformLayer_x(nn.Module):
    def __init__(self, width):
        super(DCT2DSpatialTransformLayer_x, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(width))

    def get_dct_filter(self, width):
        # dct_filter = torch.zeros(width, width, dtype=torch.float64)
        dct_filter = torch.zeros(width, width)
        for v in range(width):
            for j in range(width):
                DCT_base_x = math.cos(math.pi * (0.5 + j) * v / width) / math.sqrt(width)
                if v != 0:
                    DCT_base_x = DCT_base_x * math.sqrt(2)
                dct_filter[v, j] = DCT_base_x

        return dct_filter

    def forward(self, x):
        dct_components = []

        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, 1, x.shape[3]).expand_as(x)
            dct_components.append(dct_component.sum(3).unsqueeze(3))

        result = torch.concat(dct_components, dim=3)

        return result


class FastDCT2DSpatialTransformLayer_x(nn.Module):
    def __init__(self, width):
        super(FastDCT2DSpatialTransformLayer_x, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(width))

    def get_dct_filter(self, width):
        # dct_filter = torch.zeros(width, width, dtype=torch.float64)
        dct_filter = torch.zeros(width, width)
        for v in range(width):
            for j in range(width):
                DCT_base_x = math.cos(math.pi * (0.5 + j) * v / width) / math.sqrt(width)
                if v != 0:
                    DCT_base_x = DCT_base_x * math.sqrt(2)
                dct_filter[v, j] = DCT_base_x

        return dct_filter

    def forward(self, input):
        result = F.conv2d(input, self.weight.unsqueeze(2).unsqueeze(3))
        return result


class IDCT2DSpatialTransformLayer_x(nn.Module):
    def __init__(self, width):
        super(IDCT2DSpatialTransformLayer_x, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(width))

    def get_dct_filter(self, width):
        # dct_filter = torch.zeros(width, width, dtype=torch.float64)
        dct_filter = torch.zeros(width, width)
        for v in range(width):
            for j in range(width):
                DCT_base_x = math.cos(math.pi * (0.5 + v) * j / width) / math.sqrt(width)
                if j != 0:
                    DCT_base_x = DCT_base_x * math.sqrt(2)
                dct_filter[v, j] = DCT_base_x

        return dct_filter

    def forward(self, x):
        dct_components = []

        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, 1, x.shape[3]).expand_as(x)
            dct_components.append(dct_component.sum(3).unsqueeze(3))

        result = torch.concat(dct_components, dim=3)

        return result


class FastIDCT2DSpatialTransformLayer_x(nn.Module):
    def __init__(self, width):
        super(FastIDCT2DSpatialTransformLayer_x, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(width))

    def get_dct_filter(self, width):
        # dct_filter = torch.zeros(width, width, dtype=torch.float64)
        dct_filter = torch.zeros(width, width)
        for v in range(width):
            for j in range(width):
                DCT_base_x = math.cos(math.pi * (0.5 + v) * j / width) / math.sqrt(width)
                if j != 0:
                    DCT_base_x = DCT_base_x * math.sqrt(2)
                dct_filter[v, j] = DCT_base_x

        return dct_filter

    def forward(self, input):
        result = F.conv2d(input, self.weight.unsqueeze(2).unsqueeze(3))
        return result


class DCT2DSpatialTransformLayer_y(nn.Module):
    def __init__(self, height):
        super(DCT2DSpatialTransformLayer_y, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(height))

    def get_dct_filter(self, height):
        # dct_filter = torch.zeros(height, height, dtype=torch.float64)
        dct_filter = torch.zeros(height, height)
        for k in range(height):
            for i in range(height):
                DCT_base_y = math.cos(math.pi * (0.5 + i) * k / height) / math.sqrt(height)
                if k != 0:
                    DCT_base_y = DCT_base_y * math.sqrt(2)
                dct_filter[k, i] = DCT_base_y

        return dct_filter

    def forward(self, x):
        dct_components = []

        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, x.shape[2], 1).expand_as(x)
            dct_components.append(dct_component.sum(2).unsqueeze(2))

        result = torch.concat(dct_components, dim=2)

        return result


class FastDCT2DSpatialTransformLayer_y(nn.Module):
    def __init__(self, height):
        super(FastDCT2DSpatialTransformLayer_y, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(height))

    def get_dct_filter(self, height):
        # dct_filter = torch.zeros(height, height, dtype=torch.float64)
        dct_filter = torch.zeros(height, height)
        for k in range(height):
            for i in range(height):
                DCT_base_y = math.cos(math.pi * (0.5 + i) * k / height) / math.sqrt(height)
                if k != 0:
                    DCT_base_y = DCT_base_y * math.sqrt(2)
                dct_filter[k, i] = DCT_base_y

        return dct_filter

    def forward(self, input):
        result = F.conv2d(input, self.weight.unsqueeze(2).unsqueeze(3))
        return result


class IDCT2DSpatialTransformLayer_y(nn.Module):
    def __init__(self, height):
        super(IDCT2DSpatialTransformLayer_y, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(height))

    def get_dct_filter(self, height):
        # dct_filter = torch.zeros(height, height, dtype=torch.float64)
        dct_filter = torch.zeros(height, height)
        for k in range(height):
            for i in range(height):
                DCT_base_y = math.cos(math.pi * (0.5 + k) * i / height) / math.sqrt(height)
                if i != 0:
                    DCT_base_y = DCT_base_y * math.sqrt(2)
                dct_filter[k, i] = DCT_base_y

        return dct_filter

    def forward(self, x):
        dct_components = []

        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, x.shape[2], 1).expand_as(x)
            dct_components.append(dct_component.sum(2).unsqueeze(2))

        result = torch.concat(dct_components, dim=2)

        return result


class FastIDCT2DSpatialTransformLayer_y(nn.Module):
    def __init__(self, height):
        super(FastIDCT2DSpatialTransformLayer_y, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(height))

    def get_dct_filter(self, height):
        # dct_filter = torch.zeros(height, height, dtype=torch.float64)
        dct_filter = torch.zeros(height, height)
        for k in range(height):
            for i in range(height):
                DCT_base_y = math.cos(math.pi * (0.5 + k) * i / height) / math.sqrt(height)
                if i != 0:
                    DCT_base_y = DCT_base_y * math.sqrt(2)
                dct_filter[k, i] = DCT_base_y

        return dct_filter

    def forward(self, input):
        result = F.conv2d(input, self.weight.unsqueeze(2).unsqueeze(3))
        return result

class GroupAttentionlayer(nn.Module):
    def __init__(self, channel, groups):
        super(GroupAttentionlayer, self).__init__()
        self.fc1 = nn.Conv1d(channel, channel, 1, groups=groups)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channel, channel, 1, groups=groups)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        b,c,h,w = x.size()
        y = (x.mean(1) + x.max(1)[0]).view(b,h*w,1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y).view(b, 1, h, w).expand_as(x)
        return y * x

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

class SpatialFCAttentionlayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SpatialFCAttentionlayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        b,c,h,w = x.size()
        y = (x.mean(1) + x.max(1)[0]).view(b,h*w)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y)
        y = y.view(b, 1, h, w).expand_as(x)
        return y * x

class groupspatiallayer(nn.Module):
    def __init__(self, channel, groups):
        super(groupspatiallayer, self).__init__()
        self.fc1 = nn.Conv1d(channel, channel, 1, groups=groups)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        b,c,h,w = x.size()
        y = (x.mean(1) + x.max(1)[0]).view(b,h*w,1)
        y = self.fc1(y)
        y = self.sig(y)
        y = y.view(b, 1, h, w).expand_as(x)
        return y * x
    
class SElayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SElayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b,c,h,w = x.size()
        y = x.mean(dim=[2,3])
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y)
        x = y.unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x
    

class DCTDenoAttention(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.dct = DCT2DSpatialTransformLayer(width, height)
        self.fc = SelectGroupFClayer(width*height)
        self.idct = IDCT2DSpatialTransformLayer(width, height)

    def forward(self, x):
        if x.shape[2] != self.dct.dct_y.weight.shape[0] or x.shape[3] != self.dct.dct_y.weight.shape[1]:
            # interpolate if the size does not match
            ori_size = x.shape[2:4]
            x = F.interpolate(x, (self.dct.dct_y.weight.shape[0], self.dct.dct_y.weight.shape[1]))
            x = self.dct(x)
            x = self.fc(x)
            x = self.idct(x)
            x = F.interpolate(x, ori_size)
            return x
        x = self.dct(x)
        x = self.fc(x)
        x = self.idct(x)
        return x