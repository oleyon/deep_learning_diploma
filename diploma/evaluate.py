from pathlib import Path
import sys
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import yaml
from residual_upsampler import ResidualUpsampler
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5 import QtCore

class ImageUpscalerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        os.chdir('diploma')
    
        with open('config.yaml', 'r') as yamlfile:
            config = yaml.safe_load(yamlfile)
        model_path = Path(config['model']['path'])
        model_name = config['model']['name']
        dataset_path = config['dataset']['data_path']
        in_channels = config['model']['in_channels']
        hidden_layers = config['model']['hidden_layers']
        res_blocks_number = config['model']['res_blocks_number']
        upsample_factor = config['model']['upsample_factor']
        model_ext = config['model']['extension']
    
        model_full_path = model_path / (model_name + model_ext)


        self.setWindowTitle("Image Upscaler")

        self.image_label = QLabel(self)
        self.result_label = QLabel(self)

        self.open_button = QPushButton("Open Image", self)
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setFont(QFont('Arial', 12))

        self.upscale_button = QPushButton("Upscale Image", self)
        self.upscale_button.clicked.connect(self.upscale_image)
        self.upscale_button.setEnabled(False)
        self.upscale_button.setFont(QFont('Arial', 12))

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.open_button)

        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.upscale_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(result_layout)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.model = ResidualUpsampler(in_channels=in_channels,
                                       hidden_layers=hidden_layers,
                                       res_blocks_number=res_blocks_number,
                                       upsample_factor=upsample_factor)
        self.model.load_state_dict(torch.load(model_full_path, map_location=torch.device('cpu')))
        self.model.eval()

    def open_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.image_path = file_path
            self.display_image(self.image_label, file_path)
            self.upscale_button.setEnabled(True)

    def upscale_image(self):
        image = Image.open(self.image_path)
        image_tensor = ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            output_tensor = self.model(image_tensor)
        output_image = ToPILImage()(output_tensor.squeeze(0).clamp(0, 1))

        temp_image_path = os.path.join(os.path.dirname(self.image_path), 'temp_upscaled_image.png')
        output_image.save(temp_image_path)
        self.display_image(self.result_label, temp_image_path)

    def display_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(500, 500, aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio, transformMode=QtCore.Qt.TransformationMode.SmoothTransformation))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageUpscalerApp()
    window.show()
    sys.exit(app.exec_())
