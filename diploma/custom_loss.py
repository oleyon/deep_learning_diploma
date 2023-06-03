from metrics import SSIM
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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
        # if self.resize:
        #     input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        #     target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
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

class VGGPerceptualLoss1(nn.Module):
    def __init__(self, feature_layers=[3, 8, 15, 22], device="cuda"):
        super(VGGPerceptualLoss1, self).__init__()
        vgg_model = models.vgg19(pretrained=True).features.to(device)
        self.vgg = nn.Sequential(*list(vgg_model.children())[:max(feature_layers)+1])
        self.feature_layers = feature_layers
        self.loss = nn.L1Loss()
        self.device = device

    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)

        input_features = self.vgg(input)
        target_features = self.vgg(target)

        perceptual_loss = 0

        for layer in self.feature_layers:
            input_feat = input_features[layer]
            target_feat = target_features[layer]
            perceptual_loss += self.loss(input_feat, target_feat)

        return perceptual_loss

class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        #with torch.inference_mode():
        vgg_hr = _forward(hr)

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
    
    def forward1(self, x):
        x = self.sub_mean(x)
        x = self.vgg(x)
        return x

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class VGGWithSSIM(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGGWithSSIM, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False
        self.loss_fn = SSIMLoss(channels=256)

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        #with torch.inference_mode():
        vgg_hr = _forward(hr)

        loss = self.loss_fn(vgg_sr, vgg_hr)

        return loss
    
    def forward1(self, x):
        x = self.sub_mean(x)
        x = self.vgg(x)
        return x


class VGGWithSSIM2(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGGWithSSIM2, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])
        
        for p in self.parameters():
            p.requires_grad = False
        self.loss_fn = nn.L1Loss()

    def forward(self, sr, hr):
        def _forward(x):
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        #with torch.inference_mode():
        vgg_hr = _forward(hr)

        loss = self.loss_fn(vgg_sr, vgg_hr)

        return loss
    
    def forward1(self, x):
        x = self.vgg(x)
        return x