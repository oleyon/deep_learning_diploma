import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm
from image_dataset import ImageDataset
from autoencoder_upscale_model import AutoencoderUpscaleModel
from my_upscale_model import UpscaleModel
from timeit import default_timer as timer
from pathlib import Path
import os
import torch.nn.functional as F

    
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def downsample_image(image):
    # Calculate the downsampled size
    height, width = image.shape[2:]
    new_height, new_width = height // 2, width // 2

    # Downsample the image using bilinear interpolation
    downsampled_image = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)

    return downsampled_image

def main():
    #print(os.getcwd())
    # Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
    MODEL_PATH = Path("diploma/models")
    MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                    exist_ok=True # if models directory already exists, don't error
    )

    # Create model save path
    #MODEL_NAME = "autoencoder_upscale.pth"
    MODEL_NAME = "my_upscale.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME







    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # Set the path to the folder containing your unlabeled images
    train_path = "diploma/data/DIV2K/train_HR"
    test_path = "diploma/data/DIV2K/valid_HR"

    # Define the transformation to be applied to each image
    transform = transforms.Compose([
        #transforms.Resize((256, 256)),  # Resize the image to a fixed size
        transforms.RandomCrop((512,512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()  # Convert the image to a tensor
    ])

    # Create the ImageFolder dataset
    train_data = ImageDataset(train_path, transform=transform)
    test_data = ImageDataset(test_path, transform=transform)
    
    train_data_loader = DataLoader(dataset=train_data, 
                              batch_size=8, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

    test_data_loader = DataLoader(dataset=test_data, 
                                batch_size=8, # how many samples per batch?
                                num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=False) # shuffle the data?
    
    #model = AutoencoderUpscaleModel()
    model = UpscaleModel()
    
    if MODEL_SAVE_PATH.exists():
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
    #         f=MODEL_SAVE_PATH)
    #return
    loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001)
    
    train_time_start_on_gpu = timer()

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_data_loader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_step(data_loader=test_data_loader,
            model=model,
            loss_fn=loss_fn,
            device=device
        )

    train_time_end_on_gpu = timer()
    total_train_time_model = print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)
    
    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
            f=MODEL_SAVE_PATH)
    
    

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               #accuracy_fn = None,
               device: torch.device = "cpu"):
    train_loss = 0
    model.train()
    model.to(device)
    for batch, y in tqdm.tqdm(enumerate(data_loader)):
        # Send data to GPU
        X = downsample_image(y)
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss:.5f}")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              device: torch.device = "cpu"):
    test_loss = 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for y in data_loader:
            X = downsample_image(y)
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        print(f"Test loss: {test_loss:.5f}")

if __name__ == '__main__':
    main()