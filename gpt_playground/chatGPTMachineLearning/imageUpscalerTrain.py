import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

torch.cuda.empty_cache()

# Define the neural network architecture
class UpscaleNet(nn.Module):
    def __init__(self, num_filters=32):
        super(UpscaleNet, self).__init__()

        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(num_filters * 4, num_filters * 2, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(num_filters * 2, 3, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = nn.functional.interpolate(out, scale_factor=2, mode='bicubic', align_corners=False)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.pixel_shuffle(self.conv6(out)))
        out = F.relu(self.conv7(out))
        out = self.conv8(out)
        out = F.interpolate(out, scale_factor=2, mode='bicubic', align_corners=False)
        return out



# Load the dataset and apply transformations
dataset = datasets.ImageFolder(root='./dataset', transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
]))

# Define the data loader
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Define the model, loss function, and optimizer
model = UpscaleNet(num_filters=8)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
target_size = (256, 256) # set the target size to 256x256

# Move the model and data to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(loader):
        # Move the data to GPU
        images = images.to(device)
        
        # Upscale the images by a factor of 2
        images_upscaled = nn.functional.interpolate(images, scale_factor=2)

        # Forward pass
        outputs = model(images_upscaled)
        loss = criterion(outputs, images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'upscale_net.pth')
