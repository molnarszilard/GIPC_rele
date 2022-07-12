"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn


# ALL encoders should be called Enccoder<Model>
def get_encoder_hl():
    return eval("EncoderHigh")

def get_encoder_ll():
    return eval("EncoderLow")

class EncoderHigh(nn.Module):
    def __init__(self, img_size, latent_dim=100):
        super(EncoderHigh, self).__init__()
        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels*4, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)
        print("x0:")
        print(x.shape)
        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        print("x1:")
        print(x.shape)
        x = torch.relu(self.conv2(x))
        print("x2:")
        print(x.shape)
        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        print("x3:")
        print(x.shape)
        x = torch.relu(self.lin1(x))
        print("x4:")
        print(x.shape)
        x = torch.relu(self.lin2(x))
        print("x5:")
        print(x.shape)

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        
        return mu, logvar

class EncoderLow(nn.Module):
    def __init__(self, img_size, latent_dim=10):
        super(EncoderLow, self).__init__()
        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        if self.img_size[1] == self.img_size[2] == 128:
            self.conv_128a = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_128b = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))
        if self.img_size[1] == self.img_size[2] == 128:
            x = torch.relu(self.conv_128a(x))
            x = torch.relu(self.conv_128b(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        
        return mu, logvar