import torch
from torch import nn as nn
from functools import partial

from iharm.model.modeling.basic_blocks import ConvBlock,GaussianSmoothing
from iharm.model.ops import FeaturesConnector


class UNetEncoder(nn.Module):
    def __init__(
            self,
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels=None, backbone_mode='',_in_channels = 4
    ):
        super(UNetEncoder, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        self.block_channels = []
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]
        relu = partial(nn.ReLU, inplace=True)

        in_channels = _in_channels
        out_channels = ch

        # print(in_channels,out_channels)

        self.block0 = UNetDownBlock(
            in_channels, out_channels,
            norm_layer=norm_layer if batchnorm_from == 0 else None,
            activation=relu,
            pool=True, padding=1,
        )
        self.block_channels.append(out_channels)
        in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
        self.block1 = UNetDownBlock(
            in_channels, out_channels,
            norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None,
            activation=relu,
            pool=True, padding=1,
        )
        self.block_channels.append(out_channels)

        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels
            self.blocks_connected[f'block{block_i}'] = UNetDownBlock(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                activation=relu, padding=1,
                pool=block_i < depth - 1,
            )
            self.block_channels.append(out_channels)

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]
        outputs = []

        block_input = x
        output, block_input = self.block0(block_input)
        outputs.append(output)
        output, block_input = self.block1(block_input)
        outputs.append(output)

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                block_input = connector(block_input, stage_features)
            output, block_input = block(block_input)
            outputs.append(output)

        return outputs[::-1]


class UNetDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer,
                 attention_layer=None, attend_from=3, image_fusion=False, control_module_start = -1,
                 control_module_layer=None,
                 w_dim = 256):
        super(UNetDecoder, self).__init__()
        self.up_blocks = nn.ModuleList()
        self.image_fusion = image_fusion
        self.w_dim = w_dim
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        # Last encoder layer doesn't pool, so there're only (depth - 1) deconvs
        for d in range(depth - 1):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            use_control_module = True if control_module_start == -1 else d >= control_module_start
            stage_attention_layer = attention_layer if 0 <= attend_from <= d else None
            self.up_blocks.append(UNetUpBlock(
                in_channels, out_channels, out_channels,
                norm_layer=norm_layer, activation=partial(nn.ReLU, inplace=True),
                padding=1,
                attention_layer=stage_attention_layer,
                use_control_module = use_control_module,
                w_dim = self.w_dim,
                control_module_layer = control_module_layer,
            ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, input_image, mask, ws):
        output = encoder_outputs[0]
        count = 0
        for block, skip_output in zip(self.up_blocks, encoder_outputs[1:]):
            output = block(output, skip_output, mask,ws)
            count += 1

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * input_image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)

        return output


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, pool, padding):
        super(UNetDownBlock, self).__init__()
        self.convs = UNetDoubleConv(
            in_channels, out_channels,
            norm_layer=norm_layer, activation=activation, padding=padding,
        )
        self.pooling = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        conv_x = self.convs(x)
        return conv_x, self.pooling(conv_x)


class UNetUpBlock(nn.Module):
    def __init__(
            self,
            in_channels_decoder, in_channels_encoder, out_channels,
            norm_layer, activation, padding,
            attention_layer,
            use_control_module,
            w_dim,
            control_module_layer,
    ):
        super(UNetUpBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(
                in_channels_decoder, out_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=None, activation=activation,
            )
        )
        self.convs = UNetDoubleConv(
            in_channels_encoder + out_channels, out_channels,
            norm_layer=norm_layer, activation=activation, padding=padding,
            )
        if attention_layer is not None:
            self.attention = attention_layer(in_channels_encoder + out_channels, norm_layer, activation)
        else:
            self.attention = None

        self.w_dim = w_dim
        self.use_control_module = use_control_module
        self.out_channel = out_channels

        if self.use_control_module:
            self.control_module = control_module_layer(
                w_dim = self.w_dim,
                feature_dim = out_channels,
            )


    def forward(self, x, encoder_out, mask, ws):
        upsample_x = self.upconv(x)

        _mask = nn.functional.interpolate(
            mask, size=encoder_out.size()[-2:],
            mode='bilinear', align_corners=True
        )

        x_cat_encoder = torch.cat([encoder_out, upsample_x], dim=1)
        if self.attention is not None:
            x_cat_encoder = self.attention(x_cat_encoder, mask)
        dec_out = self.convs(x_cat_encoder)
        if self.use_control_module:
            wl, wa, wb = ws['w_l'], ws['w_a'], ws['w_b']
            dec_out = self.control_module(dec_out, wl, wa, wb, _mask)
        return dec_out


class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, padding):
        super(UNetDoubleConv, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=padding,
                norm_layer=norm_layer, activation=activation,
            ),
            ConvBlock(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=padding,
                norm_layer=norm_layer, activation=activation,
            ),
        )

    def forward(self, x):
        return self.block(x)
