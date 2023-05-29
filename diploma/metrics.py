import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class PSNR(nn.Module):
    def __init__(self, max_value=1.0):
        super(PSNR, self).__init__()
        self.max_value = max_value

    def forward(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        psnr = 20 * torch.log10(self.max_value / torch.sqrt(mse))
        return psnr

class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.window = self.create_window(window_size, channels)

    def create_window(self, window_size, channels):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * self.sigma ** 2)) for x in range(window_size)])
        window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        window = window / window.sum()
        window = window.unsqueeze(0).unsqueeze(0)
        return window.expand(channels, 1, window_size, window_size)

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channels) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return torch.mean(ssim_map)
