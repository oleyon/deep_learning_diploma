import csv
import json

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

    def save_to_csv(self, filename):
        data = {'Epoch': range(1, len(self.epoch_loss) + 1),
                'Loss': self.epoch_loss,
                'Training Time': self.epoch_training_time,
                'SSIM': self.ssim,
                'PSNR': self.psnr}

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writeheader()
            writer.writerows(data)

    def save_to_json(self, filename):
        data = {'Epoch': range(1, len(self.epoch_loss) + 1),
                'Loss': self.epoch_loss,
                'Training Time': self.epoch_training_time,
                'SSIM': self.ssim,
                'PSNR': self.psnr}

        with open(filename, 'w') as jsonfile:
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