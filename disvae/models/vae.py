"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

from disvae.utils.initialization import weights_init
from .encoders import get_encoder_hl
from .encoders import get_encoder_ll
from .decoders import get_decoder_hl
from .decoders import get_decoder_ll

def init_specific_model(img_size, latent_dim):
    """Return an instance of a VAE with encoder and decoder."""
    encoder_hl = get_encoder_hl()
    encoder_ll = get_encoder_ll()
    decoder_hl = get_decoder_hl()
    decoder_ll = get_decoder_ll()
    model = VAE(img_size, encoder_hl, encoder_ll, decoder_hl, decoder_ll, latent_dim)
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder_hl, encoder_ll, decoder_hl, decoder_ll, latent_dim):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()
        # print(img_size)
        # print(img_size[1:])
        if list(img_size[1:]) not in [[32, 32], [64, 64], [128, 128]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder_hl = encoder_hl(img_size, self.latent_dim*10)
        self.decoder_hl = decoder_hl(img_size, self.latent_dim*10)
        self.encoder_ll = encoder_ll(img_size, self.latent_dim)
        self.decoder_ll = decoder_ll(img_size, self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        print("data:")
        print(x.shape)
        latent_dist1 = self.encoder_hl(x)
        print("ld1:")
        print(latent_dist1)
        latent_sample1 = self.reparameterize(*latent_dist1)
        print("ls1:")
        print(latent_sample1.shape)
        latent_dist2 = self.encoder_ll(latent_sample1)
        print("ld2:")
        print(latent_dist2)
        latent_sample2 = self.reparameterize(*latent_dist2)
        print("ls2:")
        print(latent_sample2.shape)
        reconstruct2 = self.decoder_ll(latent_sample2)
        print("reco2:")
        print(reconstruct2.shape)
        reconstruct1 = self.decoder_hl(latent_sample1+reconstruct2)
        print("reco1:")
        print(reconstruct1.shape)
        return reconstruct1, reconstruct2, latent_dist1, latent_dist2, latent_sample1,  latent_sample2

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample
