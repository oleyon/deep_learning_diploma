import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class VGGLoss(nn.Module):
    def __init__(self, feature_layers=35, device="cpu"):
        super(VGGLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:feature_layers].to(device)
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
        input_normalized = (input - self.normalization_mean) / self.normalization_std
        input_features = self.vgg(input_normalized)

        return input_features
