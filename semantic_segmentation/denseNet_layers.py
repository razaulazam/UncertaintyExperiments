import torch.nn as nn
import torch
from helper_functions_denseNet import crop_image


class LayersDenseBlock(nn.Sequential):
    """Set of 3 layers inside the dense block."""
    def __init__(self, input_channels, growth_rate):
        super().__init__()
        self.growth_rate = growth_rate
        self.input_channels = input_channels

        self.add_module('batchnorm', nn.BatchNorm2d(self.input_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv2d', nn.Conv2d(in_channels=self.input_channels, out_channels=self.growth_rate,
                                            kernel_size=3, stride=1, padding=1))
        self.add_module('dropout',  nn.Dropout(p=0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    """Dense Block in the DenseNet Neural Network"""
    def __init__(self, input_channels, growth_rate, n_layers, up_sample=False):
        super().__init__()
        self.input_channels = input_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.up_sample = up_sample

        self.layers = nn.ModuleList([LayersDenseBlock(self.input_channels+i*self.growth_rate, self.growth_rate)
                                     for i in range(self.n_layers)])

    def forward(self, x):
        if self.up_sample:
            conv_layer_outputs = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat((x, out), 1)
                conv_layer_outputs.append(out)
            return torch.cat(conv_layer_outputs, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x


class TransitionDown(nn.Sequential):
    """Transition Down Blocks in DenseNet which consists of 1x1 convolution and pooling layers"""
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels

        self.add_module('batchnorm', nn.BatchNorm2d(input_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv2d', nn.Conv2d(in_channels=self.input_channels,out_channels=self.input_channels,
                                            kernel_size=1, stride=1, padding=0))
        self.add_module('dropout', nn.Dropout(p=0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class BottleNeck(nn.Sequential):
    """Bottle Neck layers in DenseNet which are used in the Dense Block for compressing the feature maps"""
    def __init__(self, input_channels, growth_rate, n_layers):
        super().__init__()
        self.input_channels = input_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers

        self.add_module('bottleneck', DenseBlock(self.input_channels, self.growth_rate, self.n_layers, up_sample=True))

    def forward(self, x):
        return super().forward(x)


class CompressionBlock(nn.Sequential):
    """Layer used after the Dense Block for reducing the feature dimensions"""
    def __init__(self, input_channels, compression_ratio):
        super().__init__()
        assert 0 < compression_ratio <= 1
        self.input_channels = input_channels
        self.compression_ratio = compression_ratio
        self.output_channels = int(self.compression_ratio*self.input_channels)

        self.add_module('compressionlayer', nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels,
                                                      kernel_size=1, stride=1))
        self.add_module('dropout', nn.Dropout(0.2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    """Block which is used in the up-sampling path for increasing the spatial dimensions of the feature maps"""
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.transposed_convolution = nn.ConvTranspose2d(in_channels=self.input_channels,
                                                         out_channels=self.output_channels, kernel_size=3, stride=2)

    def forward(self, x, skip):
        out = self.transposed_convolution(x)
        out = crop_image(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)

        return out






