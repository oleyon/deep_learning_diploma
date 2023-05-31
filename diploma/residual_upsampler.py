from math import log2
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super(ResBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=in_channels)
        self.prelu = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.prelu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = out + x
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super(UpsampleBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels*4,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_2 = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        #self.bn = nn.BatchNorm2d(num_features=in_channels)
        #self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.pixel_shuffle(x)
        x = self.conv_2(x)
        return x

class ResidualUpsampler(nn.Module):
    def __init__(self, in_channels=3, hidden_layers=64, res_blocks_number=4, upsample_factor=4):
        super(ResidualUpsampler, self).__init__()
        
        self.res_blocks_number = res_blocks_number
        self.upsample_factor = upsample_factor
        
        self.start_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=hidden_layers,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_layers) for _ in range(self.res_blocks_number)])
        self.upsample_blocks = nn.Sequential(*[UpsampleBlock(hidden_layers) for _ in range(int(log2(self.upsample_factor)))])
        
        self.final_conv = nn.Conv2d(in_channels=hidden_layers,
                                    out_channels=in_channels,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)

    def forward(self, x):
        x = self.start_conv(x)
        res = self.res_blocks(x)
        res = res + x
        res = self.upsample_blocks(res)
        res = self.final_conv(res)
        return res
