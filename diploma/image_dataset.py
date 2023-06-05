import os
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, upsample_factor=2):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.image_filenames[index])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        return image

class UpsampleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, upsample_factor=2):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(root_dir)
        self.upsample_factor = upsample_factor

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.image_filenames[index])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        return (self._downsample_image(image), image)
    
    def _downsample_image(self, image):
        # Calculate the downsampled size
        height, width = image.shape[1:]
        new_height, new_width = height // self.upsample_factor, width // self.upsample_factor

        # Downsample the image using bilinear interpolation
        downsampled_image = F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze()

        return downsampled_image