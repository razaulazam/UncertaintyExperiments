import torch.nn as nn
import torch
import torch.nn.functional as F
from denseNet_layers import TransitionDown, TransitionUp, BottleNeck, DenseBlock


class DenseNet(nn.Module):
    """103 layer Tiramisu Implementation"""
    def __init__(self, input_channels=3, td_blocks=(4, 5, 7, 10, 12), tu_blocks=(12, 10, 7, 5, 4), bottle_neck_layers=15,
                 growth_rate=16, first_conv_output_channels=48, num_classes=19):

        super().__init__()
        self.input_channels = input_channels
        self.td_blocks = td_blocks
        self.tu_blocks = tu_blocks
        self.bottle_neck_layers = bottle_neck_layers
        self.growth_rate = growth_rate
        self.first_conv_output_channels = first_conv_output_channels
        self.num_classes = num_classes

        self.add_module('firstconvlayer',
                        nn.Conv2d(in_channels=self.input_channels, out_channels=self.first_conv_output_channels,
                                  kernel_size=3,
                                  stride=1, padding=1))
        """Down-sampling Path"""

        self.current_channel_count = self.first_conv_output_channels

        self.denseBlocksDown = nn.ModuleList([])
        self.transitionDown = nn.ModuleList([])

        self.skip_connection_channels = []

        for i in range(len(self.td_blocks)):
            self.denseBlocksDown.append(DenseBlock(self.current_channel_count, self.growth_rate, self.td_blocks[i]))
            self.current_channel_count += self.growth_rate*self.td_blocks[i]
            self.skip_connection_channels.insert(0, self.current_channel_count)
            self.transitionDown.append(TransitionDown(self.current_channel_count))

        """Bottle-neck Layer"""

        self.add_module('bottleneck', BottleNeck(self.current_channel_count, self.growth_rate, self.bottle_neck_layers))
        self.previous_block_channels = self.growth_rate*self.bottle_neck_layers

        self.current_channel_count += self.previous_block_channels

        """Up-Sampling Path"""

        self.transitionUpBlocks = nn.ModuleList([])
        self.DenseUpBlocks = nn.ModuleList([])

        for i in range(len(tu_blocks)):
            self.transitionUpBlocks.append(TransitionUp(self.previous_block_channels, self.previous_block_channels))
            self.current_channel_count = self.previous_block_channels + self.skip_connection_channels[i]
            self.DenseUpBlocks.append(DenseBlock(self.current_channel_count, self.growth_rate, self.tu_blocks[i],
                                                 up_sample=True))
            self.previous_block_channels = self.growth_rate*self.tu_blocks[i]

        self.current_channel_count = self.previous_block_channels
        """Final Convolutional Layer"""
        output_channels_last_conv = self.num_classes*2
        self.add_module('finalconv', nn.Conv2d(in_channels=self.current_channel_count,
                                               out_channels=output_channels_last_conv, kernel_size=1, stride=1))

    def forward(self, x):
        out = self.firstconvlayer(x)
        skip_connection_tensors = []

        for i in range(len(self.td_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connection_tensors.append(out)
            out = self.transitionDown[i](out)

        out = self.bottleneck(out)

        for i in range(len(self.tu_blocks)):
            skip = skip_connection_tensors.pop()
            out = self.transitionUpBlocks[i](out, skip)
            out = self.DenseUpBlocks[i](out)

        out = self.finalconv(out)
        out = torch.split(out, self.num_classes, 1)
        semantic_result = out[0]
        aleatoric_result = F.softplus(out[1])
        out = [semantic_result, aleatoric_result]
        out = torch.cat([i.unsqueeze(1) for i in out], 1)

        return out






