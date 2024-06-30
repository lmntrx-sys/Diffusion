""" 
Defining a model for DDPM: 

    DDPM: https://arxiv.org/abs/2006.11239

    Model:  The research paper reccommends a U-net model
    First, we will define the encoder block used in the contraction path
    The second part is the decoder block, which takes the feature map from the lower layer, 
    upconverts it, crops and concatenates it with the encoder data of the same level,
    and then performs two 3×3 convolution layers followed by ReLU activation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):

    def __init__(self, inputs, num_filters):
        """
        Constructor for the Encoder class.

        Args:
            inputs (int): The number of input channels for the encoder.
            num_filters (int): The number of filters for the convolutional layers.

        Returns:
            None
        """
        super(Encoder, self).__init__()

        # Convolution with 3x3 filter followed by ReLU activation
        self.conv1 = nn.Conv2d(inputs, num_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.maxpool(x)

        return x
    
class Decoder(nn.Module):

    def __init__(self, inputs, num_filters):
        """
        Initializes a Decoder object with the given inputs and num_filters.

        Args:
            inputs (int): The number of input channels for the decoder.
            num_filters (int): The number of filters for the convolutional layers.

        Returns:
            None
        """
        super(Decoder, self).__init__()

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Convolution with 3x3 filter followed by ReLU activation
        self.conv1 = nn.Conv2d(inputs, num_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        return x
    
class UNet(nn.Module):

    def __init__(self, num_filters):
        """
        Initializes a UNet model with the specified number of filters.

        Args:
            num_filters (int): The number of filters to use in the model.

        """
        super(UNet, self).__init__()

        self.encoder1 = Encoder(3, num_filters)
        self.encoder2 = Encoder(num_filters, num_filters * 2)
        self.encoder3 = Encoder(num_filters * 2, num_filters * 4)
        self.encoder4 = Encoder(num_filters * 4, num_filters * 8)

        self.decoder1 = Decoder(num_filters * 8, num_filters * 4)
        self.decoder2 = Decoder(num_filters * 4, num_filters * 2)
        self.decoder3 = Decoder(num_filters * 2, num_filters)
        self.decoder4 = Decoder(num_filters, num_filters)

        self.final_conv = nn.Conv2d(num_filters, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Decoder
        dec1 = self.decoder1(enc4)
        dec2 = self.decoder2(dec1)
        dec3 = self.decoder3(dec2)
        dec4 = self.decoder4(dec3)

        out = self.final_conv(dec4)

        return out