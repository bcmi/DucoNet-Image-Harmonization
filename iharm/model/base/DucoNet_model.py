import torch
from functools import partial

from torch import nn as nn

from iharm.model.modeling.basic_blocks import ConvBlock, GaussianSmoothing
from iharm.model.modeling.unet_v1 import UNetEncoder, UNetDecoder
from iharm.model.ops import ChannelAttention
from iharm.model.modeling.styleganv2 import ResidualBlock
from .Control_encoder import Control_encoder


class DucoNet_model(nn.Module):
    def __init__(
            self,
            depth,
            norm_layer=nn.BatchNorm2d, batchnorm_from=2,
            attend_from=3, attention_mid_k=2.0,
            image_fusion=False,
            ch=64, max_channels=512,
            backbone_from=-1, backbone_channels=None, backbone_mode='',
            control_module_start = -1,
            lab_encoder_mask_channel=1,
            w_dim = 256,

    ):
        super(DucoNet_model, self).__init__()
        self.depth = depth
        self.control_module_start = control_module_start
        self.w_dim = w_dim

        self.lab_encoder_mask_channel = lab_encoder_mask_channel

        self.encoder = UNetEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = UNetDecoder(
            depth, self.encoder.block_channels,
            norm_layer,
            attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
            attend_from=attend_from,
            image_fusion=image_fusion,
            control_module_start=self.control_module_start,
            w_dim = self.w_dim,
            control_module_layer=Control_Module,
        )

        self.l_encoder = Control_encoder(
            depth=depth, ch=ch,  batchnorm_from=2,
            lab_encoder_mask_channel=1,
            L_or_Lab_dim=1,w_dim= self.w_dim
        )

        self.a_encoder = Control_encoder(
            depth=depth, ch=ch, batchnorm_from=2,
            lab_encoder_mask_channel=1,
            L_or_Lab_dim=1, w_dim=self.w_dim
        )

        self.b_encoder = Control_encoder(
            depth=depth, ch=ch, batchnorm_from=2,
            lab_encoder_mask_channel=1,
            L_or_Lab_dim=1, w_dim=self.w_dim
        )


    def forward(self, image, mask, image_lab=None, backbone_features=None):

        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)

        w_l = self.l_encoder(image_lab[:,0,:,:].unsqueeze(1),mask)
        w_a = self.a_encoder(image_lab[:,1,:,:].unsqueeze(1),mask)
        w_b = self.b_encoder(image_lab[:,2,:,:].unsqueeze(1),mask)

        ws = {'w_l':w_l,'w_a':w_a,'w_b':w_b}

        output = self.decoder(intermediates, image, mask, ws)

        return {'images': output}


class Control_Module(nn.Module):
    def __init__(self, w_dim, feature_dim):
        super(Control_Module,self).__init__()

        self.w_dim = w_dim
        self.feature_dim = feature_dim

        self.l_styleblock = ResidualBlock(self.w_dim, self.feature_dim, self.feature_dim)
        self.a_styleblock = ResidualBlock(self.w_dim, self.feature_dim, self.feature_dim)
        self.b_styleblock = ResidualBlock(self.w_dim, self.feature_dim, self.feature_dim)

        self.G_weight = nn.Sequential(
            nn.Conv2d(self.feature_dim * 3, 32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=True),
        )


    def forward(self, x, sl, sa, sb, mask):

        f_l = self.l_styleblock(x, sl, noise=None)
        f_a = self.a_styleblock(x, sa, noise=None)
        f_b = self.b_styleblock(x, sb, noise=None)

        f_lab = torch.cat((f_l, f_a, f_b), dim=1)
        f_weight = self.G_weight(f_lab)
        f_weight = nn.functional.softmax(f_weight, dim=1)

        weight_l = f_weight[:, 0, :, :].unsqueeze(1)
        weight_a = f_weight[:, 1, :, :].unsqueeze(1)
        weight_b = f_weight[:, 2, :, :].unsqueeze(1)

        out = weight_l * f_l + weight_a * f_a + weight_b * f_b

        out = x * (1 - mask) + mask * out
        return out

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
