import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm
import yaml
from autoencoder_upsampler import AutoencoderUpsampler
from metrics import PSNR, SSIM
from train_statistics import TrainingStatisticsLogger
from vgg_loss import *
from image_dataset import ImageDataset
from autoencoder_upscale_model import AutoencoderUpscaleModel
from my_upscale_model import UpscaleModel
from timeit import default_timer as timer
from pathlib import Path
import os
import torch.nn.functional as F
from my_upscale_model2 import UpscaleModel2
from custom_loss import *

    
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def downsample_image(image, factor=2):
    # Calculate the downsampled size
    height, width = image.shape[2:]
    new_height, new_width = height // factor, width // factor

    # Downsample the image using bilinear interpolation
    downsampled_image = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)

    return downsampled_image

def main():
    with open('diploma/config.yaml', 'r') as yamlfile:
        config = yaml.safe_load(yamlfile)

    model_path = Path(config['model']['path'])
    learning_rate = config['train']['learning_rate']
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    model_name = config['model']['name']
    dataset_path = config['dataset']['data_path']
    train_path = dataset_path + '/train'
    test_path = dataset_path + '/valid'
    device = config['train']['device']
    #upsample_factor = config['model']['upsample_factor']
    weight_decay = config['train']['optimizer']['weight_decay']
    log_dir = config['logging']['log_dir']
    
    model_path.parent.mkdir(parents=True,
                    exist_ok=True)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transformation to be applied to each image
    transform = transforms.Compose([
        transforms.RandomCrop(config['dataset']['transform'][1]['RandomCrop']['size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Create the ImageFolder dataset
    train_data = ImageDataset(train_path, transform=transform)
    test_data = ImageDataset(test_path, transform=transform)
    
    train_data_loader = DataLoader(dataset=train_data, 
                              batch_size=batch_size,
                              num_workers=1,
                              shuffle=True) 

    test_data_loader = DataLoader(dataset=test_data, 
                                batch_size=batch_size,
                                num_workers=1,
                                shuffle=False)

    model = AutoencoderUpsampler()

    if model_path.exists():
        print("loading existing model")
        model.load_state_dict(torch.load(model_path))
    
    #loss_fn = VGGPerceptualLoss().to(device)
    #loss_fn = nn.MSELoss()
    #loss_fn = SSIMLoss()
    loss_fn = CombinedLoss(loss_shift=1)
    psnr = PSNR()
    ssim = SSIM()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    
    train_statistics_logger = TrainingStatisticsLogger()
    test_statistics_logger = TrainingStatisticsLogger()
    
    train_time_start_on_gpu = timer()

    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_data_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            psnr=psnr,
            ssim=ssim,
            logger=train_statistics_logger
        )
        test_step(data_loader=test_data_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            psnr=psnr,
            ssim=ssim,
            logger=test_statistics_logger
        )

    train_time_end_on_gpu = timer()
    total_train_time_model = print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)

    # Save the model state dict and statistics
    print(f"Saving model to: {model_path}")
    train_statistics_logger.save_to_json(log_dir + model_name + '_train_log.json')
    train_statistics_logger.save_to_csv(log_dir + model_name + '_train_log.csv')
    test_statistics_logger.save_to_json(log_dir + model_name + '_test_log.json')
    test_statistics_logger.save_to_csv(log_dir + model_name + '_test_log.csv')
    torch.save(obj=model.state_dict(),
            f=model_path)
    
    

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               logger: TrainingStatisticsLogger,
               ssim,
               psnr,
               device: torch.device = "cpu"):
    train_loss = 0
    psnr_acc = 0
    ssim_acc = 0
    model.train()
    model.to(device)
    loss_fn.to(device)
    #ssim.to(device).eval()
    start_time = timer()
    for batch, y in tqdm.tqdm(enumerate(data_loader)):
        # Send data to GPU
        X = downsample_image(y, factor=4)
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate loss
        loss = loss_fn(y_pred, y)
        
        psnr_acc += psnr(y_pred, y).item()
        
        ssim_acc += ssim(y_pred, y).item()

        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    end_time = timer()
    total_time = end_time - start_time
    train_loss /= len(data_loader)
    psnr_acc /= len(data_loader)
    ssim_acc /= len(data_loader)
    
    logger.log_statistics(loss=train_loss, epoch_duration=total_time, ssim=ssim_acc, psnr=psnr_acc)
    
    print(f"Train loss: {train_loss:.5f}, SSIM: {ssim_acc:.5f}, PSNR: {psnr_acc:.5f}")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              logger: TrainingStatisticsLogger,
              ssim,
              psnr,
              device: torch.device = "cpu"):
    test_loss = 0
    psnr_acc = 0
    ssim_acc = 0
    model.to(device)
    loss_fn.to(device)
    #ssim.to(device).eval()
    #psnr.to(device).eval()
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        start = timer()
        for y in data_loader:
            X = downsample_image(y, factor=4)
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            test_pred = model(X)
            
            # Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)

            psnr_acc += psnr(test_pred, y).item()
            
            ssim_acc += ssim(test_pred, y).item()
        
        end= timer()
        # Adjust metrics and print out
        total_time = end - start
        test_loss /= len(data_loader)
        psnr_acc /= len(data_loader)
        ssim_acc /= len(data_loader)

        logger.log_statistics(loss=test_loss, epoch_duration=total_time, ssim=ssim_acc, psnr=psnr_acc)
        
        print(f"Test loss: {test_loss:.5f}, SSIM: {ssim_acc:.5f}, PSNR: {psnr_acc:.5f}")

if __name__ == '__main__':
    main()