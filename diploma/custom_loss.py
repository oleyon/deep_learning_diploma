from metrics import SSIM
import torch
import torch.nn as nn
import torchvision.models as models

class SSIMLoss(SSIM):
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super(SSIMLoss, self).__init__(window_size, sigma, channels)

    def forward(self, input, target):
        ssim_val = super().forward(input, target)
        ssim_loss = (1 - ssim_val) / 2
        return ssim_loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, ssim_window_size=11, ssim_sigma=1.5, ssim_channels=3, vgg_resize=True, loss_shift=0.5) -> None:
        super(CombinedLoss, self).__init__()
        self.ssim_loss = SSIMLoss(ssim_window_size, ssim_sigma, ssim_channels)
        self.vgg_loss = VGGPerceptualLoss(vgg_resize)
        self.loss_shift = loss_shift
    
    def forward(self, input, target):
        ssim_loss = self.ssim_loss(input, target)
        vgg_loss = self.vgg_loss(input, target)
        return vgg_loss + self.loss_shift * ssim_loss
