import torch
from functools import partial

from torch import nn as nn

from iharm.model.modeling.basic_blocks import ConvBlock, GaussianSmoothing
from iharm.model.modeling.unet_v1 import UNetEncoder, UNetDecoder
from iharm.model.ops import ChannelAttention
from iharm.model.modeling.styleganv2 import ResidualBlock


class Control_encoder(nn.Module):
    def __init__(
            self,
            depth,
            norm_layer=nn.BatchNorm2d, batchnorm_from=2,
            ch=64, max_channels=512,
            lab_encoder_mask_channel=1,
            L_or_Lab_dim=3,w_dim=256,
            backbone_from=-1, backbone_channels=None, backbone_mode='',
    ):
        super(Control_encoder, self).__init__()
        self.depth = depth
        self.w_dim = w_dim
        self.lab_encoder_mask_channel = lab_encoder_mask_channel

        self.lab_encoder = UNetEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode,
            _in_channels=L_or_Lab_dim+self.lab_encoder_mask_channel
        )

        self.map2w = map_net(int_dim=256,out_dim=self.w_dim)


    def forward(self, image_lab, mask,backbone_features=None):
        x_lab = image_lab
        if self.lab_encoder_mask_channel==1:
            x_lab = torch.cat((x_lab, mask), dim=1)
        intermediates_lab = self.lab_encoder(x_lab,backbone_features)
        w = self.map2w(intermediates_lab[0],mask)

        return w

class map_net(nn.Module):
    def __init__(self,int_dim = 256,out_dim = 256):
        super(map_net, self).__init__()

        self.int_dim = int_dim
        self.pooling = nn.AdaptiveAvgPool2d((1,1))

        self.net = nn.Sequential(
            nn.Linear(int_dim,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,out_dim),
        )

    def forward(self,feature_map,mask):

        w_bg = self.pooling(feature_map).view(-1,self.int_dim)

        return self.net(w_bg)


class SpatialSeparatedAttention(nn.Module):
    def __init__(self, in_channels, norm_layer, activation, mid_k=2.0):
        super(SpatialSeparatedAttention, self).__init__()
        self.background_gate = ChannelAttention(in_channels)
        self.foreground_gate = ChannelAttention(in_channels)
        self.mix_gate = ChannelAttention(in_channels)

        mid_channels = int(mid_k * in_channels)
        self.learning_block = nn.Sequential(
            ConvBlock(
                in_channels, mid_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
            ConvBlock(
                mid_channels, in_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
        )
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)

    def forward(self, x, mask):
        mask = self.mask_blurring(nn.functional.interpolate(
            mask, size=x.size()[-2:],
            mode='bilinear', align_corners=True
        ))
        background = self.background_gate(x)
        foreground = self.learning_block(self.foreground_gate(x))
        mix = self.mix_gate(x)
        output = mask * (foreground + mix) + (1 - mask) * background
        return output
