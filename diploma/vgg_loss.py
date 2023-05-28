import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class VGGLoss(nn.Module):
    def __init__(self, feature_layers=35, device="cpu"):
        super(VGGLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:22].to(device)
        self.loss = nn.MSELoss()

        # Set the model to evaluation mode
        self.vgg.eval()
        
        # Freeze all the parameters in the VGG model
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.device = device

        # Create a normalization transform to preprocess the input images
        #self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        #self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        # Normalize the input and target images
        #input_normalized = (input - self.normalization_mean) / self.normalization_std
        #target_normalized = (target - self.normalization_mean) / self.normalization_std
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        # vgg_loss = 0
        # for layer_id in self.layer_ids:
        #     input_feat = input_features[layer_id]
        #     target_feat = target_features[layer_id]

        #     vgg_loss += self.loss(input_feat, target_feat)
        
        vgg_loss = self.loss(input_features, target_features)

        return vgg_loss
    
    
    def forward_demo(self, input):
        input = input.to(self.device)
        # Normalize the input and target images
        #input_normalized = (input - self.normalization_mean) / self.normalization_std
        input_features = self.vgg(input)

        return input_features



class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[3, 8, 15, 22], device="cpu"):
        super(VGGPerceptualLoss, self).__init__()
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
    

class VGGPerceptualLoss_1(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss_1, self).__init__()
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