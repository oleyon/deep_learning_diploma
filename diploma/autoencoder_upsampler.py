from typing import Any
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
        self.prelu = nn.PReLU()
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
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels,
                                           out_channels=in_channels//2,
                                           kernel_size=3,
                                           padding=1,
                                           output_padding=1,
                                           stride=2)
        self.conv = nn.Conv2d(in_channels=in_channels//2,
                              out_channels=in_channels//2,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn = nn.BatchNorm2d(num_features=in_channels//2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.upsample(x)
        #print(x.shape)
        x = self.relu(self.bn(self.conv(x)))
        return x



class AutoencoderUpsampler(nn.Module):
    def __init__(self, in_channels=3, hidden_layers=64):
        super(AutoencoderUpsampler, self).__init__()
        
        self.start_conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=hidden_layers,
                                kernel_size=9,
                                stride=1,
                                padding=4)
        self.max_pool = nn.MaxPool2d(2)
        self.res_1 = ResBlock(hidden_layers)
        self.res_2 = ResBlock(hidden_layers)
        self.conv_1 = nn.Conv2d(in_channels=hidden_layers,
                                out_channels=hidden_layers*2,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.res_3 = ResBlock(hidden_layers*2)
        self.conv_2 = nn.Conv2d(in_channels=hidden_layers*2,
                                out_channels=hidden_layers*4,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.res_4 = ResBlock(hidden_layers*4)
        self.upsample_1 = UpsampleBlock(hidden_layers*4)
        self.upsample_2 = UpsampleBlock(hidden_layers*2)
        self.upsample_3 = UpsampleBlock(hidden_layers*2)
        self.upsample_4 = UpsampleBlock(hidden_layers*2)
        self.conv_u_3 = nn.Conv2d(in_channels=hidden_layers,
                                  out_channels=hidden_layers*2,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2)
        self.conv_u_4 = nn.Conv2d(in_channels=hidden_layers,
                                  out_channels=hidden_layers*2,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2)
        self.final_res = ResBlock(hidden_layers)
        self.final_conv = nn.Conv2d(in_channels=hidden_layers,
                                    out_channels=in_channels,
                                    kernel_size=9,
                                    stride=1,
                                    padding=4)
        #self.activation = nn.Tanh()

    def forward(self, x):
        encoded = self.start_conv(x) # [8, 64, 64, 64]
        encoded = self.res_1(encoded) # [8, 64, 64, 64]
        downsample_1 = self.max_pool(self.res_2(encoded)) # [8, 64, 32, 32]
        downsample_1 = self.conv_1(downsample_1) # [8, 128, 32, 32]
        downsample_2 = self.max_pool(self.res_3(downsample_1)) # [8, 128, 16, 16]
        downsample_2 = self.conv_2(downsample_2) # [8, 256, 16, 16]
        downsample_2 = self.res_4(downsample_2) # [8, 256, 16, 16]
        upsample = self.upsample_1(downsample_2) # [8, 128, 32, 32]
        upsample = upsample + downsample_1 # [8, 128, 32, 32] + [8, 128, 32, 32]
        upsample = self.upsample_2(upsample)
        upsample = upsample + encoded
        upsample = self.conv_u_3(upsample)
        upsample = self.upsample_3(upsample)
        upsample = self.conv_u_4(upsample)
        upsample = self.upsample_4(upsample)
        upsample = self.final_res(upsample)
        upsample = self.final_conv(upsample)
        return upsample
