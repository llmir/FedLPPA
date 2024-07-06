# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function
from re import X
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, output_size, img_size, hidden_sizes=[256, 128, 64, 32], negative_slope=0.01):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.img_size = img_size

        # 创建隐藏层和批量归一化层
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            out_size = hidden_sizes[i]
            self.hidden_layers.append(nn.Linear(in_size, out_size))
            # self.hidden_layers.append(nn.BatchNorm1d(out_size))  # 批量归一化
            self.hidden_layers.append(nn.LeakyReLU(negative_slope=negative_slope))

        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size)
            # nn.BatchNorm2d(output_size)
            # nn.LeakyReLU(negative_slope=negative_slope)
            )
            

    def forward(self, x):
        # 前插值
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x = x.transpose(1,3).transpose(1,2)  # 重构输入
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc_out(x)
        return x.transpose(1,3).transpose(2,3)

class MLP_BN(nn.Module):
    def __init__(self, input_size, output_size, img_size, hidden_sizes=[128, 64], negative_slope=0.01):
        super(MLP_BN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.img_size = img_size
        self.hidden_sizes = hidden_sizes

        # 创建隐藏层和批量归一化层
        self.hidden_linear_layers = nn.ModuleList()
        self.hidden_bn_layers = nn.ModuleList()
        self.hidden_leakyrelu_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            out_size = hidden_sizes[i]
            self.hidden_linear_layers.append(nn.Linear(in_size, out_size))
            self.hidden_bn_layers.append(nn.BatchNorm2d(out_size))  # 批量归一化
            self.hidden_leakyrelu_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
    


        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size)
            # nn.BatchNorm2d(output_size)
            # nn.LeakyReLU(negative_slope=negative_slope)
            )

        def forward(self, x):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            x = x.transpose(1,3).transpose(1,2)  # 重构输入
            for i in range(len(self.hidden_sizes)):
                x = self.hidden_linear_layers[i](x)
                x = x.transpose(1,3).transpose(2,3)
                x = self.hidden_bn_layers[i](x)
                x = self.hidden_leakyrelu_layers[i](x)
                x = x.transpose(1,3).transpose(1,2)
            x = self.fc_out(x)
            return x.transpose(1,3).transpose(2,3)
            
class MLP_BN_Later_Interpolate(nn.Module):
    def __init__(self, input_size, output_size, img_size, hidden_sizes=[128, 64], negative_slope=0.01):
        super(MLP_BN_Later_Interpolate, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.img_size = img_size
        self.hidden_sizes = hidden_sizes

        # 创建隐藏层和批量归一化层
        self.hidden_linear_layers = nn.ModuleList()
        self.hidden_bn_layers = nn.ModuleList()
        self.hidden_leakyrelu_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            out_size = hidden_sizes[i]
            self.hidden_linear_layers.append(nn.Linear(in_size, out_size))
            self.hidden_bn_layers.append(nn.BatchNorm2d(out_size))  # 批量归一化
            self.hidden_leakyrelu_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
    
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size)

            )
            

    def forward(self, x):
        x = x.transpose(1,3).transpose(1,2)  # 重构输入
        for i in range(len(self.hidden_sizes)):
            x = self.hidden_linear_layers[i](x)
            x = x.transpose(1,3).transpose(2,3)
            x = self.hidden_bn_layers[i](x)
            x = self.hidden_leakyrelu_layers[i](x)
            x = x.transpose(1,3).transpose(1,2)
        x = self.fc_out(x)
        x = x.transpose(1,3).transpose(2,3)
        # 后插值
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return x


class MLP_BN_GAP(nn.Module):
    def __init__(self, input_size, output_size, img_size, hidden_sizes=[128, 64], negative_slope=0.01):
        super(MLP_BN_GAP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.img_size = img_size
        self.hidden_sizes = hidden_sizes

        # 创建隐藏层和批量归一化层
        self.hidden_linear_layers = nn.ModuleList()
        self.hidden_bn_layers = nn.ModuleList()
        self.hidden_leakyrelu_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            out_size = hidden_sizes[i]
            self.hidden_linear_layers.append(nn.Linear(in_size, out_size))
            self.hidden_bn_layers.append(nn.BatchNorm2d(out_size))  # 批量归一化
            self.hidden_leakyrelu_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
    
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size)

            )
            

    def forward(self, x):
        x = x.transpose(1,3).transpose(1,2)  # 重构输入
        for i in range(len(self.hidden_sizes)):
            x = self.hidden_linear_layers[i](x)
            x = x.transpose(1,3).transpose(2,3)
            x = self.hidden_bn_layers[i](x)
            x = self.hidden_leakyrelu_layers[i](x)
            x = x.transpose(1,3).transpose(1,2)
        x = self.fc_out(x)
        x = x.transpose(1,3).transpose(2,3)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        # 后插值
        # x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return x
        
    

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class PersonalizedChannelSelection(nn.Module):
    def __init__(self, f_dim, emb_dim):
        super(PersonalizedChannelSelection, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Sequential(nn.Conv2d(emb_dim, f_dim, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(f_dim, f_dim, 1, bias=False))
        self.fc2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(f_dim // 16, f_dim, 1, bias=False))

    def forward_emb(self, emb):
        emb = emb.unsqueeze(-1).unsqueeze(-1)
        emb = self.fc1(emb)
        return emb

    def forward(self, x, emb):
        b, c, w, h = x.size()

        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        # site embedding
        emb = self.forward_emb(emb)

        # avg
        avg_out = torch.cat([avg_out, emb], dim=1)
        avg_out = self.fc2(avg_out)

        # max
        max_out = torch.cat([max_out, emb], dim=1)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        hmap = self.sigmoid(out)

        x = x * hmap + x

        return x, hmap

class LCEncoder(nn.Module):
    def __init__(self, params):
        super(LCEncoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        self.n_pcs = self.params['pcs_num']
        self.n_emb = self.params['emb_num']
        self.n_client = self.params['client_num']
        self.cid = self.params['client_id']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.conv_list = [self.in_conv, self.down1, self.down2, self.down3, self.down4]

        self.pcs_list = []
        for i in range(self.n_pcs): 
            # print('PCS', (2**(5 - self.n_pcs + i)) * 16, self.ft_chns[5 - self.n_pcs + i])
            self.pcs_list.append(
                PersonalizedChannelSelection(self.ft_chns[5 - self.n_pcs + i], self.n_emb).cuda()
            )

    def forward(self, x, emb_idx=None):
        def get_emb(emb_idx):
            batch_size = x.size(0)
            emb = torch.zeros((batch_size, self.n_client)).cuda()
            emb[:, emb_idx] = 1
            return emb

        if not emb_idx:
            emb = get_emb(self.cid)
        else:
            emb = get_emb(emb_idx)

        heatmaps = []
        features = []
        for i in range(len(self.conv_list)):
            # print(i, x.shape, self.conv_list[i])
            x = self.conv_list[i](x)
            if i >= (len(self.conv_list) - self.n_pcs):
                x, heatmap = self.pcs_list[i - len(self.conv_list) + self.n_pcs](x, emb)
            else:
                heatmap = None
            features.append(x)
            heatmaps.append(heatmap)

        return features, heatmaps


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        output = self.out_conv(x_4)
        return output, x_1, x_2, x_3, x_4

class Decoder_MLP(nn.Module):
    def __init__(self, params):
        super(Decoder_MLP, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature, prototypes):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        x_4 = x_4 * prototypes
        output = self.out_conv(x_4)
        return output, x_1, x_2, x_3, x_4

class Decoder_320(nn.Module):
    def __init__(self, params):
        super(Decoder_320, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            320, self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        output = self.out_conv(x_4)
        return output, x_1, x_2, x_3, x_4

class Decoder_Head(nn.Module):
    def __init__(self, params):
        super(Decoder_Head, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(self.ft_chns[2], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        output = self.out_conv(x_4)
        aux_output = self.dsn_head(x_2)
        return output, x_1, x_2, x_3, x_4, aux_output


class Decoder_MultiHead(nn.Module):
    def __init__(self, params):
        super(Decoder_MultiHead, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.dsn_head1 = nn.Sequential(
            nn.Conv2d(self.ft_chns[2], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.dsn_head2 = nn.Sequential(
            nn.Conv2d(self.ft_chns[1], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.dsn_head3 = nn.Sequential(
            nn.Conv2d(self.ft_chns[0], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        output = self.out_conv(x_4)
        aux_output1 = self.dsn_head1(x_2)
        aux_output2 = self.dsn_head2(x_3)
        aux_output3 = self.dsn_head3(x_4)
        return output, x_1, x_2, x_3, x_4, aux_output1, aux_output2, aux_output3

class Decoder_MultiHead_Two(nn.Module):
    def __init__(self, params):
        super(Decoder_MultiHead_Two, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.dsn_head1 = nn.Sequential(
            nn.Conv2d(self.ft_chns[2], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.dsn_head2 = nn.Sequential(
            nn.Conv2d(self.ft_chns[1], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        output = self.out_conv(x_4)
        aux_output1 = self.dsn_head1(x_2)
        aux_output2 = self.dsn_head2(x_3)
        return output, x_1, x_2, x_3, x_4, aux_output1, aux_output2


class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder_URDS(nn.Module):
    def __init__(self, params):
        super(Decoder_URDS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, de1, de2, de3, de4 = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4]

class UNet_320(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_320, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_320(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, de1, de2, de3, de4 = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4]


class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)[0]
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)[0]
        return main_seg, aux_seg1


class UNet_CCT_3H(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT_3H, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [FeatureNoise()(i) for i in feature]
        aux_seg2 = self.aux_decoder1(aux2_feature)
        return main_seg, aux_seg1, aux_seg2


class UNet_Head(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_Head, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_Head(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, de1, de2, de3, de4, aux_output = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, aux_output]


class UNet_MultiHead(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_MultiHead, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_MultiHead(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, de1, de2, de3, de4, aux_output1, aux_output2, aux_output3 = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, aux_output1, aux_output2, aux_output3]


class UNet_Uni(nn.Module):
    def __init__(self, in_chns, class_num, client_num, client_id):
        super(UNet_Uni, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        decoder_params = copy.deepcopy(params)
        decoder_params['feature_chns'][-1] += 1
        self.decoder = Decoder(decoder_params)

        fuse_conv_list = [
            nn.Conv2d(params['feature_chns'][-1] + client_num, params['feature_chns'][-2], kernel_size=3, padding=1),
            nn.BatchNorm2d(params['feature_chns'][-2]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(params['feature_chns'][-2], params['feature_chns'][-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(params['feature_chns'][-3]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(params['feature_chns'][-3], client_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(client_num),
            nn.LeakyReLU()
        ]

        self.fuse = nn.Sequential(*fuse_conv_list)
        self.client_num = client_num
        self.client_id = client_id

    def forward(self, x):
        feature = self.encoder(x)
        n, c, h, w = feature[-1].shape
        if not hasattr(self, 'uni_prompt'):
            self.uni_prompt = nn.Parameter(
                torch.rand(size=(self.client_num, h, w), device=feature[-1].device),
            requires_grad=True)
        feature_c_list = [
            torch.cat([feature[-1][i, :, :, :], self.uni_prompt], dim=0).unsqueeze(0)
            for i in range(n)
        ]
        fuse_feature = torch.cat(feature_c_list, dim=0)
        prompts = self.fuse(fuse_feature)
        client_prompt = prompts[:, self.client_id].unsqueeze(1)
        feature[-1] = torch.cat([feature[-1], client_prompt], dim=1)
        output, de1, de2, de3, de4 = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, prompts, feature[-1]]


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//8,kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out

class AffinityAttentionConcat(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttentionConcat, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        # print("sab.shape=",sab.shape)
        cab = self.cab(x)
        # print("cab.shape=",cab.shape)
        # out = sab + cab  #[b,64,h,w]
        out = torch.cat((sab, cab), dim=1)#[b,128,h,w]

        # out = torch.cat((cab, sab), dim=1)#[b,128,h,w]
        # print("sab concat cab.shape=",out.shape)
        return out


class MultiAttention(nn.Module):

    def __init__(self, embed_dim):
        super(MultiAttention, self).__init__()
        self.multi = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.size()
        multi = x.view(B, -1).unsqueeze(1)
        out = self.multi(query=multi, key=multi, value=multi).reshape(B, C, H, W)
        # print(out.shape)
        return out


class UNet_UniV2(nn.Module):
    def __init__(self, in_chns, class_num, prompt_type, attention_type, sup_type, use_label_prompt, client_num, client_id):
        super(UNet_UniV2, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        decoder_params = copy.deepcopy(params)
        decoder_params['feature_chns'][-1] += params['feature_chns'][-3]
        self.decoder = Decoder(decoder_params)
        self.decoder_auxiliary = Decoder(decoder_params)

        self.label_types = ['scribble', 'block', 'keypoint']
        assert sup_type in self.label_types
        assert use_label_prompt in [0, 1]
        if use_label_prompt:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + client_num + len(self.label_types), params['feature_chns'][-2], kernel_size=3, padding=1)]
        else:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + client_num, params['feature_chns'][-2], kernel_size=3, padding=1)]
        fuse_conv_list += [
            nn.BatchNorm2d(params['feature_chns'][-2]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(params['feature_chns'][-2], params['feature_chns'][-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(params['feature_chns'][-3]),
            nn.LeakyReLU()
        ]
        self.fuse = nn.Sequential(*fuse_conv_list)
        assert attention_type in ['dual', 'sab', 'cab', 'multi']
        if attention_type == 'dual':
            self.attention = AffinityAttention(params['feature_chns'][-3])
        elif attention_type == 'sab':
            self.attention = SpatialAttentionBlock(params['feature_chns'][-3])
        elif attention_type == 'cab':
            self.attention = ChannelAttentionBlock(params['feature_chns'][-3])
        if sup_type == 'box':
            self.sup_type = 'block'
        elif sup_type == 'scribble_noisy':
            self.sup_type = 'scribble'
        else:
            self.sup_type = sup_type
        self.prompt_type = prompt_type
        self.attention_type = attention_type
        # self.sup_type = sup_type
        self.use_label_prompt = use_label_prompt
        self.client_num = client_num
        self.client_id = client_id

    def forward(self, x):
        feature = self.encoder(x)
        n, c, h, w = feature[-1].shape
        if not hasattr(self, 'uni_prompt'):
            assert self.prompt_type in ['universal', 'onehot']
            if self.prompt_type == 'universal':
                self.uni_prompt = nn.Parameter(
                    torch.rand(size=(self.client_num, h, w), device=feature[-1].device),
                requires_grad=True)
            elif self.prompt_type == 'onehot':
                prompt_list = []
                for i in range(self.client_num):
                    if self.client_id == i:
                        prompt_temp = torch.ones(size=(1, h, w), device=feature[-1].device)
                    else:
                        prompt_temp = torch.zeros(size=(1, h, w), device=feature[-1].device)
                    prompt_list.append(prompt_temp)
                self.uni_prompt = torch.cat(prompt_list, dim=0)

        if (self.use_label_prompt == 1) and (not hasattr(self, 'label_prompt')):
            prompt_list = []
            for type_temp in self.label_types:
                if self.sup_type == type_temp:
                    prompt_temp = torch.ones(size=(1, h, w), device=feature[-1].device)
                else:
                    prompt_temp = torch.zeros(size=(1, h, w), device=feature[-1].device)
                prompt_list.append(prompt_temp)
            self.label_prompt = torch.cat(prompt_list, dim=0)

        if self.attention_type == 'multi' and (not hasattr(self, 'attention')):
            self.attention = MultiAttention(embed_dim=c * h * w)

        if self.use_label_prompt == 1:
            feature_c_list = [
                torch.cat([feature[-1][i, :, :, :], self.uni_prompt, self.label_prompt], dim=0).unsqueeze(0)
                for i in range(n)
            ]
        else:
            feature_c_list = [
                torch.cat([feature[-1][i, :, :, :], self.uni_prompt], dim=0).unsqueeze(0)
                for i in range(n)
            ]
        fuse_feature = torch.cat(feature_c_list, dim=0)
        fuse_feature = self.fuse(fuse_feature)
        fuse_feature = self.attention(fuse_feature)
        feature[-1] = torch.cat([feature[-1], fuse_feature], dim=1)
        output, de1, de2, de3, de4 = self.decoder(feature)
        output_auxiliary, _, _, _, _ = self.decoder_auxiliary(feature)
        return [output, feature, de1, de2, de3, de4, self.uni_prompt, feature[-1], output_auxiliary]


class UNet_UniV3(nn.Module):
    def __init__(self, in_chns, class_num, prompt_type, attention_type, sup_type, use_label_prompt, client_num, client_id, img_size):
        super(UNet_UniV3, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        decoder_params = copy.deepcopy(params)
        decoder_params['feature_chns'][-1] += params['feature_chns'][-3]
        self.decoder = Decoder(decoder_params)
        self.decoder_auxiliary = Decoder(decoder_params)

        self.label_types = ['scribble', 'block', 'keypoint']
        # assert sup_type in self.label_types
        assert use_label_prompt in [0, 1]
        if use_label_prompt:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + client_num + len(self.label_types), params['feature_chns'][-2], kernel_size=3, padding=1)]
        else:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + client_num, params['feature_chns'][-2], kernel_size=3, padding=1)]
        fuse_conv_list += [
            nn.BatchNorm2d(params['feature_chns'][-2]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(params['feature_chns'][-2], params['feature_chns'][-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(params['feature_chns'][-3]),
            nn.LeakyReLU()
        ]
        self.fuse = nn.Sequential(*fuse_conv_list)
        assert attention_type in ['dual', 'sab', 'cab', 'multi']
        if attention_type == 'dual':
            self.attention = AffinityAttention(params['feature_chns'][-3])
        elif attention_type == 'sab':
            self.attention = SpatialAttentionBlock(params['feature_chns'][-3])
        elif attention_type == 'cab':
            self.attention = ChannelAttentionBlock(params['feature_chns'][-3])
        if sup_type == 'box':
            self.sup_type = 'block'
        elif sup_type == 'scribble_noisy':
            self.sup_type = 'scribble'
        else:
            self.sup_type = sup_type
        self.prompt_type = prompt_type
        self.attention_type = attention_type
        # self.sup_type = sup_type
        self.use_label_prompt = use_label_prompt
        self.client_num = client_num
        self.client_id = client_id
        
        assert self.prompt_type in ['universal', 'onehot']
        if self.prompt_type == 'universal':
            self.uni_prompt = nn.Parameter(
                torch.rand(size=(self.client_num, int(img_size/16), int(img_size/16))),
            requires_grad=True)
        elif self.prompt_type == 'onehot':
            prompt_list = []
            for i in range(self.client_num):
                if self.client_id == i:
                    prompt_temp = torch.ones(size=(1, int(img_size/16), int(img_size/16)))
                else:
                    prompt_temp = torch.zeros(size=(1, int(img_size/16), int(img_size/16)))
                prompt_list.append(prompt_temp)
            self.uni_prompt = torch.cat(prompt_list, dim=0)

    def forward(self, x):
        feature = self.encoder(x)
        uni_prompt = self.uni_prompt
        n, c, h, w = feature[-1].shape
        if (self.use_label_prompt == 1) and (not hasattr(self, 'label_prompt')):
            prompt_list = []
            for type_temp in self.label_types:
                if self.sup_type == type_temp:
                    prompt_temp = torch.ones(size=(1, h, w), device=feature[-1].device)
                else:
                    prompt_temp = torch.zeros(size=(1, h, w), device=feature[-1].device)
                prompt_list.append(prompt_temp)
            self.label_prompt = torch.cat(prompt_list, dim=0)

        if self.attention_type == 'multi' and (not hasattr(self, 'attention')):
            self.attention = MultiAttention(embed_dim=c * h * w)

        if self.use_label_prompt == 1:
            feature_c_list = [
                torch.cat([feature[-1][i, :, :, :], uni_prompt, self.label_prompt], dim=0).unsqueeze(0)
                for i in range(n)
            ]
        else:
            feature_c_list = [
                torch.cat([feature[-1][i, :, :, :], uni_prompt], dim=0).unsqueeze(0)
                for i in range(n)
            ]
        fuse_feature = torch.cat(feature_c_list, dim=0)
        fuse_feature = self.fuse(fuse_feature)
        fuse_feature = self.attention(fuse_feature)
        feature[-1] = torch.cat([feature[-1], fuse_feature], dim=1)
        output, de1, de2, de3, de4 = self.decoder(feature)
        output_auxiliary, _, _, _, _ = self.decoder_auxiliary(feature)
        return [output, feature, de1, de2, de3, de4, uni_prompt.clone(), feature[-1], output_auxiliary]

# error
class UNet_UniV4(nn.Module):
    def __init__(self, in_chns, class_num, prompt_type, attention_type, sup_type, use_label_prompt, client_num, client_id, img_size):
        super(UNet_UniV4, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        decoder_params = copy.deepcopy(params)
        decoder_params['feature_chns'][-1] += params['feature_chns'][-3]
        self.decoder = Decoder_MLP(decoder_params)
        self.decoder_auxiliary = Decoder(decoder_params)
        self.mlp = MLP_BN_Later_Interpolate(320, params["feature_chns"][0], img_size)

        self.label_types = ['scribble', 'block', 'keypoint']
        # assert sup_type in self.label_types
        assert use_label_prompt in [0, 1]
        if use_label_prompt:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + client_num + len(self.label_types), params['feature_chns'][-2], kernel_size=3, padding=1)]
        else:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + client_num, params['feature_chns'][-2], kernel_size=3, padding=1)]
        fuse_conv_list += [
            nn.BatchNorm2d(params['feature_chns'][-2]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(params['feature_chns'][-2], params['feature_chns'][-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(params['feature_chns'][-3]),
            nn.LeakyReLU()
        ]
        self.fuse = nn.Sequential(*fuse_conv_list)
        assert attention_type in ['dual', 'sab', 'cab', 'multi']
        if attention_type == 'dual':
            self.attention = AffinityAttention(params['feature_chns'][-3])
        elif attention_type == 'sab':
            self.attention = SpatialAttentionBlock(params['feature_chns'][-3])
        elif attention_type == 'cab':
            self.attention = ChannelAttentionBlock(params['feature_chns'][-3])
        if sup_type == 'box':
            self.sup_type = 'block'
        elif sup_type == 'scribble_noisy':
            self.sup_type = 'scribble'
        else:
            self.sup_type = sup_type
        self.prompt_type = prompt_type
        self.attention_type = attention_type
        # self.sup_type = sup_type
        self.use_label_prompt = use_label_prompt
        self.client_num = client_num
        self.client_id = client_id
        
        assert self.prompt_type in ['universal', 'onehot']
        if self.prompt_type == 'universal':
            self.uni_prompt = nn.Parameter(
                torch.rand(size=(self.client_num, int(img_size/16), int(img_size/16))),
            requires_grad=True)
        elif self.prompt_type == 'onehot':
            prompt_list = []
            for i in range(self.client_num):
                if self.client_id == i:
                    prompt_temp = torch.ones(size=(1, int(img_size/16), int(img_size/16)))
                else:
                    prompt_temp = torch.zeros(size=(1, int(img_size/16), int(img_size/16)))
                prompt_list.append(prompt_temp)
            self.uni_prompt = torch.cat(prompt_list, dim=0)

    def forward(self, x):
        feature = self.encoder(x)
        uni_prompt = self.uni_prompt
        n, c, h, w = feature[-1].shape
        if (self.use_label_prompt == 1) and (not hasattr(self, 'label_prompt')):
            prompt_list = []
            for type_temp in self.label_types:
                if self.sup_type == type_temp:
                    prompt_temp = torch.ones(size=(1, h, w), device=feature[-1].device)
                else:
                    prompt_temp = torch.zeros(size=(1, h, w), device=feature[-1].device)
                prompt_list.append(prompt_temp)
            self.label_prompt = torch.cat(prompt_list, dim=0)

        if self.attention_type == 'multi' and (not hasattr(self, 'attention')):
            self.attention = MultiAttention(embed_dim=c * h * w)

        if self.use_label_prompt == 1:
            feature_c_list = [
                torch.cat([feature[-1][i, :, :, :], uni_prompt, self.label_prompt], dim=0).unsqueeze(0)
                for i in range(n)
            ]
        else:
            feature_c_list = [
                torch.cat([feature[-1][i, :, :, :], uni_prompt], dim=0).unsqueeze(0)
                for i in range(n)
            ]
        fuse_feature = torch.cat(feature_c_list, dim=0)
        fuse_feature = self.fuse(fuse_feature)
        fuse_feature = self.attention(fuse_feature)
        feature[-1] = torch.cat([feature[-1], fuse_feature], dim=1)
        # print(feature[-1].size())
        prototypes = self.mlp(feature[-1])
        output, de1, de2, de3, de4 = self.decoder(feature, prototypes)
        output_auxiliary, _, _, _, _ = self.decoder_auxiliary(feature)
        # output = output *prototypes
        return [output, feature, de1, de2, de3, de4, uni_prompt.clone(), feature[-1], output_auxiliary]



class UNet_UniV5(nn.Module):
    def __init__(self, in_chns, class_num, prompt_type, attention_type, sup_type, use_label_prompt, client_num, client_id, img_size):
        super(UNet_UniV5, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        decoder_params = copy.deepcopy(params)
        decoder_params['feature_chns'][-1] += params['feature_chns'][-3]
        self.decoder = Decoder_MLP(decoder_params)
        self.decoder_auxiliary = Decoder(decoder_params)
        self.mlp = MLP_BN_GAP(320, params["feature_chns"][0], img_size)

        self.label_types = ['scribble', 'block', 'keypoint']
        # assert sup_type in self.label_types
        assert use_label_prompt in [0, 1]
        if use_label_prompt:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + 2 + len(self.label_types), params['feature_chns'][-2], kernel_size=3, padding=1)]
        else:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + 2, params['feature_chns'][-2], kernel_size=3, padding=1)]
        fuse_conv_list += [
            nn.BatchNorm2d(params['feature_chns'][-2]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(params['feature_chns'][-2], params['feature_chns'][-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(params['feature_chns'][-3]),
            nn.LeakyReLU()
        ]
        self.fuse = nn.Sequential(*fuse_conv_list)
        assert attention_type in ['dual', 'sab', 'cab', 'multi']
        if attention_type == 'dual':
            self.attention = AffinityAttention(params['feature_chns'][-3])
        elif attention_type == 'sab':
            self.attention = SpatialAttentionBlock(params['feature_chns'][-3])
        elif attention_type == 'cab':
            self.attention = ChannelAttentionBlock(params['feature_chns'][-3])
        if sup_type == 'box':
            self.sup_type = 'block'
        elif sup_type == 'scribble_noisy':
            self.sup_type = 'scribble'
        else:
            self.sup_type = sup_type
        self.prompt_type = prompt_type
        self.attention_type = attention_type
        # self.sup_type = sup_type
        self.use_label_prompt = use_label_prompt
        self.client_num = client_num
        self.client_id = client_id
        
        assert self.prompt_type in ['universal', 'onehot']
        if self.prompt_type == 'universal':
            self.distribution_prompts = nn.Parameter(
                torch.rand(size=(self.client_num, int(img_size/16), int(img_size/16))),
            requires_grad=True)
            self.uni_prompt = nn.Parameter(
                torch.rand(size=(1, int(img_size/16), int(img_size/16))),
            requires_grad=True)
            # self.uni_prompt = torch.cat(prompt_list, dim=0)
        elif self.prompt_type == 'onehot':
            prompt_list = []
            for i in range(self.client_num):
                if self.client_id == i:
                    prompt_temp = torch.ones(size=(1, int(img_size/16), int(img_size/16)), requires_grad=False)
                else:
                    prompt_temp = torch.zeros(size=(1, int(img_size/16), int(img_size/16)), requires_grad=False)
                prompt_list.append(prompt_temp)
            self.distribution_prompts = nn.Parameter(torch.cat(prompt_list, dim=0))
            self.uni_prompt = nn.Parameter(
                torch.rand(size=(1, int(img_size/16), int(img_size/16))),
            requires_grad=True).cuda()

    def forward(self, x):
        clients_fuse = []
        fuse_features = []
        feature = self.encoder(x)
        uni_prompt = self.uni_prompt
        distribution_prompts = self.distribution_prompts
        # print("distribution_prompts",distribution_prompts.shape)
        n, c, h, w = feature[-1].shape
        if (self.use_label_prompt == 1) and (not hasattr(self, 'label_prompt')):
            prompt_list = []
            for type_temp in self.label_types:
                if self.sup_type == type_temp:
                    prompt_temp = torch.ones(size=(1, h, w), device=feature[-1].device)
                else:
                    prompt_temp = torch.zeros(size=(1, h, w), device=feature[-1].device)
                prompt_list.append(prompt_temp)
            self.label_prompt = torch.cat(prompt_list, dim=0)

        if self.attention_type == 'multi' and (not hasattr(self, 'attention')):
            self.attention = MultiAttention(embed_dim=c * h * w)

        if self.use_label_prompt == 1:
            feature_c_list = []
            for client in range(self.client_num):
                feature_client = [
                    torch.cat([feature[-1][i, :, :, :], distribution_prompts[client].unsqueeze(0), self.uni_prompt, self.label_prompt], dim=0).unsqueeze(0) for i in range(n)
                ]
                feature_c_list.append(feature_client)
            for client in range(self.client_num):
                clients_fuse.append(torch.cat(feature_c_list[client], dim=0))
        else:
            feature_c_list = []
            for client in range(self.client_num):
                feature_client = [
                    torch.cat([feature[-1][i, :, :, :], distribution_prompts[client].unsqueeze(0), self.uni_prompt], dim=0).unsqueeze(0) for i in range(n)
                ]
                feature_c_list.append(feature_client)
            for client in range(self.client_num):
                clients_fuse.append(torch.cat(feature_c_list[client], dim=0))
            
        
        for client in range(self.client_num):
            fuse_feature = self.fuse(clients_fuse[client])
            fuse_feature = self.attention(fuse_feature)
            fuse_features.append(fuse_feature)

        feature[-1] = torch.cat([feature[-1], fuse_features[self.client_id]], dim=1)
        # print(feature[-1].size())
        prototypes = self.mlp(feature[-1])
        output, de1, de2, de3, de4 = self.decoder(feature, prototypes)
        output_auxiliary, _, _, _, _ = self.decoder_auxiliary(feature)
        # output = output *prototypes
        return [output, feature, de1, de2, de3, de4, fuse_features, feature[-1], output_auxiliary, distribution_prompts.clone(), uni_prompt.clone()]

class UNet_Univ5_Ablation(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_Univ5_Ablation, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, de1, de2, de3, de4 = self.main_decoder(feature)
        aux1_feature = feature
        aux_seg1 = self.aux_decoder1(aux1_feature)[0]
        return [output, feature, de1, de2, de3, de4, aux_seg1]

class UNet_UniV5_WO_Uni_Prompt(nn.Module):
    def __init__(self, in_chns, class_num, prompt_type, attention_type, sup_type, use_label_prompt, client_num, client_id, img_size):
        super(UNet_UniV5_WO_Uni_Prompt, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        decoder_params = copy.deepcopy(params)
        decoder_params['feature_chns'][-1] += params['feature_chns'][-3]
        self.decoder = Decoder_MLP(decoder_params)
        self.decoder_auxiliary = Decoder(decoder_params)
        self.mlp = MLP_BN_GAP(320, params["feature_chns"][0], img_size)

        self.label_types = ['scribble', 'block', 'keypoint']
        # assert sup_type in self.label_types
        assert use_label_prompt in [0, 1]
        if use_label_prompt:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + 1 + len(self.label_types), params['feature_chns'][-2], kernel_size=3, padding=1)]
        else:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + 2, params['feature_chns'][-2], kernel_size=3, padding=1)]
        fuse_conv_list += [
            nn.BatchNorm2d(params['feature_chns'][-2]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(params['feature_chns'][-2], params['feature_chns'][-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(params['feature_chns'][-3]),
            nn.LeakyReLU()
        ]
        self.fuse = nn.Sequential(*fuse_conv_list)
        assert attention_type in ['dual', 'sab', 'cab', 'multi']
        if attention_type == 'dual':
            self.attention = AffinityAttention(params['feature_chns'][-3])
        elif attention_type == 'sab':
            self.attention = SpatialAttentionBlock(params['feature_chns'][-3])
        elif attention_type == 'cab':
            self.attention = ChannelAttentionBlock(params['feature_chns'][-3])
        if sup_type == 'box':
            self.sup_type = 'block'
        elif sup_type == 'scribble_noisy':
            self.sup_type = 'scribble'
        else:
            self.sup_type = sup_type
        self.prompt_type = prompt_type
        self.attention_type = attention_type
        # self.sup_type = sup_type
        self.use_label_prompt = use_label_prompt
        self.client_num = client_num
        self.client_id = client_id
        
        assert self.prompt_type in ['universal', 'onehot']
        if self.prompt_type == 'universal':
            self.distribution_prompts = nn.Parameter(
                torch.rand(size=(self.client_num, int(img_size/16), int(img_size/16))),
            requires_grad=True)
            self.uni_prompt = nn.Parameter(
                torch.rand(size=(1, int(img_size/16), int(img_size/16))),
            requires_grad=True)
            # self.uni_prompt = torch.cat(prompt_list, dim=0)
        elif self.prompt_type == 'onehot':
            prompt_list = []
            for i in range(self.client_num):
                if self.client_id == i:
                    prompt_temp = torch.ones(size=(1, int(img_size/16), int(img_size/16)))
                else:
                    prompt_temp = torch.zeros(size=(1, int(img_size/16), int(img_size/16)))
                prompt_list.append(prompt_temp)
            self.uni_prompt = torch.cat(prompt_list, dim=0)

    def forward(self, x):
        clients_fuse = []
        fuse_features = []
        feature = self.encoder(x)
        # uni_prompt = self.uni_prompt
        distribution_prompts = self.distribution_prompts
        # print("distribution_prompts",distribution_prompts.shape)
        n, c, h, w = feature[-1].shape
        if (self.use_label_prompt == 1) and (not hasattr(self, 'label_prompt')):
            prompt_list = []
            for type_temp in self.label_types:
                if self.sup_type == type_temp:
                    prompt_temp = torch.ones(size=(1, h, w), device=feature[-1].device)
                else:
                    prompt_temp = torch.zeros(size=(1, h, w), device=feature[-1].device)
                prompt_list.append(prompt_temp)
            self.label_prompt = torch.cat(prompt_list, dim=0)

        if self.attention_type == 'multi' and (not hasattr(self, 'attention')):
            self.attention = MultiAttention(embed_dim=c * h * w)

        if self.use_label_prompt == 1:
            feature_c_list = []
            for client in range(self.client_num):
                feature_client = [
                    torch.cat([feature[-1][i, :, :, :], distribution_prompts[client].unsqueeze(0), self.label_prompt], dim=0).unsqueeze(0) for i in range(n)
                ]
                feature_c_list.append(feature_client)
            for client in range(self.client_num):
                clients_fuse.append(torch.cat(feature_c_list[client], dim=0))
        else:
            feature_c_list = []
            for client in range(self.client_num):
                feature_client = [
                    torch.cat([feature[-1][i, :, :, :], distribution_prompts[client].unsqueeze(0), self.uni_prompt], dim=0).unsqueeze(0) for i in range(n)
                ]
                feature_c_list.append(feature_client)
            for client in range(self.client_num):
                clients_fuse.append(torch.cat(feature_c_list[client], dim=0))
            
        
        for client in range(self.client_num):
            fuse_feature = self.fuse(clients_fuse[client])
            fuse_feature = self.attention(fuse_feature)
            fuse_features.append(fuse_feature)

        feature[-1] = torch.cat([feature[-1], fuse_features[self.client_id]], dim=1)
        # print(feature[-1].size())
        prototypes = self.mlp(feature[-1])
        output, de1, de2, de3, de4 = self.decoder(feature, prototypes)
        output_auxiliary, _, _, _, _ = self.decoder_auxiliary(feature)
        # output = output *prototypes
        return [output, feature, de1, de2, de3, de4, fuse_features, feature[-1], output_auxiliary, distribution_prompts.clone()]

class UNet_UniV5_WO_Uni_Prompt(nn.Module):
    def __init__(self, in_chns, class_num, prompt_type, attention_type, sup_type, use_label_prompt, client_num, client_id, img_size):
        super(UNet_UniV5_WO_Uni_Prompt, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        decoder_params = copy.deepcopy(params)
        decoder_params['feature_chns'][-1] += params['feature_chns'][-3]
        self.decoder = Decoder_MLP(decoder_params)
        self.decoder_auxiliary = Decoder(decoder_params)
        self.mlp = MLP_BN_GAP(320, params["feature_chns"][0], img_size)

        self.label_types = ['scribble', 'block', 'keypoint']
        # assert sup_type in self.label_types
        assert use_label_prompt in [0, 1]
        if use_label_prompt:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + 1 + len(self.label_types), params['feature_chns'][-2], kernel_size=3, padding=1)]
        else:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + 2, params['feature_chns'][-2], kernel_size=3, padding=1)]
        fuse_conv_list += [
            nn.BatchNorm2d(params['feature_chns'][-2]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(params['feature_chns'][-2], params['feature_chns'][-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(params['feature_chns'][-3]),
            nn.LeakyReLU()
        ]
        self.fuse = nn.Sequential(*fuse_conv_list)
        assert attention_type in ['dual', 'sab', 'cab', 'multi']
        if attention_type == 'dual':
            self.attention = AffinityAttention(params['feature_chns'][-3])
        elif attention_type == 'sab':
            self.attention = SpatialAttentionBlock(params['feature_chns'][-3])
        elif attention_type == 'cab':
            self.attention = ChannelAttentionBlock(params['feature_chns'][-3])
        if sup_type == 'box':
            self.sup_type = 'block'
        elif sup_type == 'scribble_noisy':
            self.sup_type = 'scribble'
        else:
            self.sup_type = sup_type
        self.prompt_type = prompt_type
        self.attention_type = attention_type
        # self.sup_type = sup_type
        self.use_label_prompt = use_label_prompt
        self.client_num = client_num
        self.client_id = client_id
        
        assert self.prompt_type in ['universal', 'onehot']
        if self.prompt_type == 'universal':
            self.distribution_prompts = nn.Parameter(
                torch.rand(size=(self.client_num, int(img_size/16), int(img_size/16))),
            requires_grad=True)
            self.uni_prompt = nn.Parameter(
                torch.rand(size=(1, int(img_size/16), int(img_size/16))),
            requires_grad=True)
            # self.uni_prompt = torch.cat(prompt_list, dim=0)
        elif self.prompt_type == 'onehot':
            prompt_list = []
            for i in range(self.client_num):
                if self.client_id == i:
                    prompt_temp = torch.ones(size=(1, int(img_size/16), int(img_size/16)))
                else:
                    prompt_temp = torch.zeros(size=(1, int(img_size/16), int(img_size/16)))
                prompt_list.append(prompt_temp)
            self.uni_prompt = torch.cat(prompt_list, dim=0)

    def forward(self, x):
        clients_fuse = []
        fuse_features = []
        feature = self.encoder(x)
        # uni_prompt = self.uni_prompt
        distribution_prompts = self.distribution_prompts
        # print("distribution_prompts",distribution_prompts.shape)
        n, c, h, w = feature[-1].shape
        if (self.use_label_prompt == 1) and (not hasattr(self, 'label_prompt')):
            prompt_list = []
            for type_temp in self.label_types:
                if self.sup_type == type_temp:
                    prompt_temp = torch.ones(size=(1, h, w), device=feature[-1].device)
                else:
                    prompt_temp = torch.zeros(size=(1, h, w), device=feature[-1].device)
                prompt_list.append(prompt_temp)
            self.label_prompt = torch.cat(prompt_list, dim=0)

        if self.attention_type == 'multi' and (not hasattr(self, 'attention')):
            self.attention = MultiAttention(embed_dim=c * h * w)

        if self.use_label_prompt == 1:
            feature_c_list = []
            for client in range(self.client_num):
                feature_client = [
                    torch.cat([feature[-1][i, :, :, :], distribution_prompts[client].unsqueeze(0), self.label_prompt], dim=0).unsqueeze(0) for i in range(n)
                ]
                feature_c_list.append(feature_client)
            for client in range(self.client_num):
                clients_fuse.append(torch.cat(feature_c_list[client], dim=0))
        else:
            feature_c_list = []
            for client in range(self.client_num):
                feature_client = [
                    torch.cat([feature[-1][i, :, :, :], distribution_prompts[client].unsqueeze(0), self.uni_prompt], dim=0).unsqueeze(0) for i in range(n)
                ]
                feature_c_list.append(feature_client)
            for client in range(self.client_num):
                clients_fuse.append(torch.cat(feature_c_list[client], dim=0))
            
        
        for client in range(self.client_num):
            fuse_feature = self.fuse(clients_fuse[client])
            fuse_feature = self.attention(fuse_feature)
            fuse_features.append(fuse_feature)

        feature[-1] = torch.cat([feature[-1], fuse_features[self.client_id]], dim=1)
        # print(feature[-1].size())
        prototypes = self.mlp(feature[-1])
        output, de1, de2, de3, de4 = self.decoder(feature, prototypes)
        output_auxiliary, _, _, _, _ = self.decoder_auxiliary(feature)
        # output = output *prototypes
        return [output, feature, de1, de2, de3, de4, fuse_features, feature[-1], output_auxiliary, distribution_prompts.clone()]

class UNet_LC(nn.Module):
    def __init__(self, in_chns, class_num, pcs_num, emb_num, client_num, client_id):
        super(UNet_LC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'pcs_num': pcs_num,
                  'emb_num': emb_num,
                  'client_num': client_num,
                  'client_id': client_id}

        self.encoder = LCEncoder(params)
        self.decoder = Decoder_Head(params)

    def forward(self, x, emb_idx=None):
        feature, heatmap = self.encoder(x, emb_idx)
        output, de1, de2, de3, de4, aux_output  = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, heatmap,aux_output]
    
class UNet_LC_Auxi(nn.Module):
    def __init__(self, in_chns, class_num, pcs_num, emb_num, client_num, client_id):
        super(UNet_LC_Auxi, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'pcs_num': pcs_num,
                  'emb_num': emb_num,
                  'client_num': client_num,
                  'client_id': client_id}

        self.encoder = LCEncoder(params)
        self.decoder = Decoder_Head(params)
        self.decoder_auxiliary = Decoder(params)

    def forward(self, x, emb_idx=None):
        feature, heatmap = self.encoder(x, emb_idx)
        output, de1, de2, de3, de4, aux_output  = self.decoder(feature)
        output_auxiliary, _, _, _, _ = self.decoder_auxiliary(feature)
        return [output, feature, de1, de2, de3, de4, heatmap, aux_output, output_auxiliary]


class UNet_UniV5_AttentionConcat(nn.Module):
    def __init__(self, in_chns, class_num, prompt_type, attention_type, sup_type, use_label_prompt, client_num, client_id, img_size):
        super(UNet_UniV5_AttentionConcat, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        decoder_params = copy.deepcopy(params)
        decoder_params['feature_chns'][-1] = decoder_params['feature_chns'][-1] + 2 * params['feature_chns'][-3]
        self.decoder = Decoder_MLP(decoder_params)
        self.decoder_auxiliary = Decoder(decoder_params)
        self.mlp = MLP_BN_GAP(384, params["feature_chns"][0], img_size)

        self.label_types = ['scribble', 'block', 'keypoint']
        # assert sup_type in self.label_types
        assert use_label_prompt in [0, 1]
        if use_label_prompt:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + 2 + len(self.label_types), params['feature_chns'][-2], kernel_size=3, padding=1)]
        else:
            fuse_conv_list = [nn.Conv2d(params['feature_chns'][-1] + 2, params['feature_chns'][-2], kernel_size=3, padding=1)]
        fuse_conv_list += [
            nn.BatchNorm2d(params['feature_chns'][-2]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(params['feature_chns'][-2], params['feature_chns'][-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(params['feature_chns'][-3]),
            nn.LeakyReLU()
        ]
        self.fuse = nn.Sequential(*fuse_conv_list)
        assert attention_type in ['dual', 'sab', 'cab', 'multi', 'dual_concat']
        if attention_type == 'dual':
            self.attention = AffinityAttention(params['feature_chns'][-3])
        elif attention_type == 'sab':
            self.attention = SpatialAttentionBlock(params['feature_chns'][-3])
        elif attention_type == 'cab':
            self.attention = ChannelAttentionBlock(params['feature_chns'][-3])
        elif attention_type == 'dual_concat':
            self.attention = AffinityAttentionConcat(params['feature_chns'][-3])
        if sup_type == 'box':
            self.sup_type = 'block'
        elif sup_type == 'scribble_noisy':
            self.sup_type = 'scribble'
        else:
            self.sup_type = sup_type
        self.prompt_type = prompt_type
        self.attention_type = attention_type
        # self.sup_type = sup_type
        self.use_label_prompt = use_label_prompt
        self.client_num = client_num
        self.client_id = client_id
        
        assert self.prompt_type in ['universal', 'onehot']
        if self.prompt_type == 'universal':
            self.distribution_prompts = nn.Parameter(
                torch.rand(size=(self.client_num, int(img_size/16), int(img_size/16))),
            requires_grad=True)
            self.uni_prompt = nn.Parameter(
                torch.rand(size=(1, int(img_size/16), int(img_size/16))),
            requires_grad=True)
            # self.uni_prompt = torch.cat(prompt_list, dim=0)
        elif self.prompt_type == 'onehot':
            prompt_list = []
            for i in range(self.client_num):
                if self.client_id == i:
                    prompt_temp = torch.ones(size=(1, int(img_size/16), int(img_size/16)))
                else:
                    prompt_temp = torch.zeros(size=(1, int(img_size/16), int(img_size/16)))
                prompt_list.append(prompt_temp)
            self.uni_prompt = torch.cat(prompt_list, dim=0)

    def forward(self, x):
        clients_fuse = []
        fuse_features = []
        feature = self.encoder(x)
        uni_prompt = self.uni_prompt
        distribution_prompts = self.distribution_prompts
        # print("distribution_prompts",distribution_prompts.shape)
        n, c, h, w = feature[-1].shape
        if (self.use_label_prompt == 1) and (not hasattr(self, 'label_prompt')):
            prompt_list = []
            for type_temp in self.label_types:
                if self.sup_type == type_temp:
                    prompt_temp = torch.ones(size=(1, h, w), device=feature[-1].device)
                else:
                    prompt_temp = torch.zeros(size=(1, h, w), device=feature[-1].device)
                prompt_list.append(prompt_temp)
            self.label_prompt = torch.cat(prompt_list, dim=0)

        if self.use_label_prompt == 1:
            feature_c_list = []
            for client in range(self.client_num):
                feature_client = [
                    torch.cat([feature[-1][i, :, :, :], distribution_prompts[client].unsqueeze(0), self.uni_prompt, self.label_prompt], dim=0).unsqueeze(0) for i in range(n)
                ]
                feature_c_list.append(feature_client)
            for client in range(self.client_num):
                clients_fuse.append(torch.cat(feature_c_list[client], dim=0))
        else:
            feature_c_list = []
            for client in range(self.client_num):
                feature_client = [
                    torch.cat([feature[-1][i, :, :, :], distribution_prompts[client].unsqueeze(0), self.uni_prompt], dim=0).unsqueeze(0) for i in range(n)
                ]
                feature_c_list.append(feature_client)
            for client in range(self.client_num):
                clients_fuse.append(torch.cat(feature_c_list[client], dim=0))
            
        
        for client in range(self.client_num):
            fuse_feature = self.fuse(clients_fuse[client])
            fuse_feature = self.attention(fuse_feature)
            fuse_features.append(fuse_feature)

        feature[-1] = torch.cat([feature[-1], fuse_features[self.client_id]], dim=1)
        # print(feature[-1].size())
        prototypes = self.mlp(feature[-1])
        output, de1, de2, de3, de4 = self.decoder(feature, prototypes)
        output_auxiliary, _, _, _, _ = self.decoder_auxiliary(feature)
        # output = output *prototypes
        return [output, feature, de1, de2, de3, de4, fuse_features, feature[-1], output_auxiliary, distribution_prompts.clone(), uni_prompt.clone()]


class UNet_LC_MultiHead(nn.Module):
    def __init__(self, in_chns, class_num, pcs_num, emb_num, client_num, client_id):
        super(UNet_LC_MultiHead, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'pcs_num': pcs_num,
                  'emb_num': emb_num,
                  'client_num': client_num,
                  'client_id': client_id}

        self.encoder = LCEncoder(params)
        self.decoder = Decoder_MultiHead(params)

    def forward(self, x, emb_idx=None):
        feature, heatmap = self.encoder(x, emb_idx)
        output, de1, de2, de3, de4, aux_output1, aux_output2, aux_output3  = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, heatmap, aux_output1, aux_output2, aux_output3]


class UNet_LC_MultiHead_Two(nn.Module):
    def __init__(self, in_chns, class_num, pcs_num, emb_num, client_num, client_id):
        super(UNet_LC_MultiHead_Two, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'pcs_num': pcs_num,
                  'emb_num': emb_num,
                  'client_num': client_num,
                  'client_id': client_id}

        self.encoder = LCEncoder(params)
        self.decoder = Decoder_MultiHead_Two(params)

    def forward(self, x, emb_idx=None):
        feature, heatmap = self.encoder(x, emb_idx)
        output, de1, de2, de3, de4, aux_output1, aux_output2  = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, heatmap, aux_output1, aux_output2]
