import csv
import json
from pathlib import Path

class TrainingStatisticsLogger:
    def __init__(self):
        self.epoch_loss = []
        self.epoch_training_time = []
        self.ssim = []
        self.psnr = []

    def log_statistics(self, loss, epoch_duration, ssim, psnr):
        self.epoch_loss.append(loss.item())
        self.epoch_training_time.append(epoch_duration)
        self.ssim.append(ssim)
        self.psnr.append(psnr)

    def save_to_csv(self, filename, append=False):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if append else 'w'

        data = [{'Epoch': epoch,
                 'Loss': loss,
                 'Training Time': time,
                 'SSIM': ssim,
                 'PSNR': psnr} for epoch, loss, time, ssim, psnr in
                zip(range(1, len(self.epoch_loss) + 1),
                    self.epoch_loss,
                    self.epoch_training_time,
                    self.ssim,
                    self.psnr)]

        with open(filename, mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
            if not append:
                writer.writeheader()
            writer.writerows(data)

    def save_to_json(self, filename, append=False):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if append else 'w'

        data = {'Epoch': list(range(1, len(self.epoch_loss) + 1)),
                'Loss': self.epoch_loss,
                'Training Time': self.epoch_training_time,
                'SSIM': self.ssim,
                'PSNR': self.psnr}

        if append and path.exists() and path.stat().st_size > 0:
            with open(filename, 'r') as jsonfile:
                existing_data = json.load(jsonfile)
                existing_data['Epoch'].extend(data['Epoch'])
                existing_data['Loss'].extend(data['Loss'])
                existing_data['Training Time'].extend(data['Training Time'])
                existing_data['SSIM'].extend(data['SSIM'])
                existing_data['PSNR'].extend(data['PSNR'])
            with open(filename, 'w') as jsonfile:
                json.dump(existing_data, jsonfile)
        else:
            with open(filename, mode) as jsonfile:
                json.dump(data, jsonfile)

    def load_from_csv(self, filename):
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.epoch_loss.append(float(row['Loss']))
                self.epoch_training_time.append(float(row['Training Time']))
                self.ssim.append(float(row['SSIM']))
                self.psnr.append(float(row['PSNR']))

    def load_from_json(self, filename):
        with open(filename, 'r') as jsonfile:
            data = json.load(jsonfile)
            self.epoch_loss = data['Loss']
            self.epoch_training_time = data['Training Time']
            self.ssim = data['SSIM']
            self.psnr = data['PSNR']
