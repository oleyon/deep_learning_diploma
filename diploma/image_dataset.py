import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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