import torch
import torch.nn as nn

class UpscaleModel2(nn.Module):
    def __init__(self):
        super(UpscaleModel2, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.PReLU()
            # Add more convolutional layers as needed
        )

        # # Upsampling layers
        self.upsample_layers = nn.Sequential(
            #nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
            # Add more upsampling layers as needed
        )
        
        #self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        # Final convolutional layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.upsample_layers(x)
        x = self.final_layer(x)
        return x