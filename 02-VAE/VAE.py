import torch
import os
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import DatasetFolder
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init


import torch.nn.functional as F

import itertools

import numpy as np

from tensorboardX import SummaryWriter


batch_size = 100
n_epoch = 50
lam = 1
z_dim = 128
train_path = 'data_aligned_resize/train/'
writer = SummaryWriter()


USE_CUDA = True
use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


train_dataset = ImageFolder(
        root=train_path,
        transform=torchvision.transforms.ToTensor()
    )
train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(Encoder, self).__init__()

        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*4*4)),
        )

        self.mean = nn.Linear(1024*4*4, z_dim)
        self.logvar = nn.Linear(1024*4*4, z_dim)


    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        multi = torch.FloatTensor(std.size()).normal_().cuda()
        multi = Variable(multi)
        return multi.mul(std).add_(mean)


    def forward(self, x):
        b_size = x.size(0)

        h = self.encoder(x)
        
        mean, logvar = self.mean(h), self.logvar(h)

        latent_z = self.reparametrize(mean, logvar)

        return latent_z, mean, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim=10, nc = 3):
        super(Decoder, self).__init__()
        
        self.nc = nc
        self.z_dim = z_dim

        self.decoder_lin = nn.Sequential(
            nn.Linear(z_dim, 1024*8*8),
            View((-1, 1024, 8, 8))
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1)
        )
        
    def forward(self, x):
        h = self.decoder_lin(x)
        out = self.decoder_conv(h)
        return out


class VAE(nn.Module):
    def __init__(self, z_dim=128, nc=3):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        self.encoder = Encoder(z_dim=z_dim, nc=nc)
        self.decoder = Decoder(z_dim=z_dim, nc=nc)

    
    def forward(self, x):
        latent_z, mean, logvar = self.encoder(x)
        rec_x = self.decoder(latent_z)
        return rec_x, mean, logvar
    
    def _encode(self, x):
        latent_z, _, _ = self.encoder(x)
        return latent_z
    
    def _decode(self, x):
        return self.decoder(x)



VAE_net = VAE(z_dim=128, nc=3).to(device)

vae_optim = optim.Adam(VAE_net.parameters(), lr = 2e-04, betas=(0.5, 0.999))


reconst_criterion = nn.MSELoss(reduction='sum')

def kld_criterion(mean, logvar):
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), 1)
    kld_loss = torch.sum(kld_loss)
    return kld_loss

VAE_net.train()

for epoch in range(n_epoch):
    for batch_idx,(data,_) in enumerate(train_loader):

        vae_optim.zero_grad()

        b_size = data.shape[0]

        imgs = Variable(data).to(device)

        recons_x, mean, logvar = VAE_net(imgs)

        recon_loss = reconst_criterion(imgs, recons_x)
        kld_loss = kld_criterion(mean, logvar)
        
        loss_vae = recon_loss + kld_loss

        loss_vae.backward()
        vae_optim.step()   
        
        
        print("Train Epoch: {} [{}/{} ({:.0f}%)] \tVAE_Loss: {:.6f}\tRecon_Loss: {:.6f}\tKLD_Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_vae.item(), recon_loss.item(), kld_loss.item()))

        n_epochs = epoch*len(train_loader)+batch_idx
        writer.add_scalar('Recon_loss', recon_loss.data.cpu().numpy(), n_epochs)
        writer.add_scalar('Vae_loss', loss_vae.data.cpu().numpy(), n_epochs)
        writer.add_scalar('kld_loss', kld_loss.data.cpu().numpy(), n_epochs)
        
        if batch_idx % 5 == 0:

            zx = torch.randn(batch_size, z_dim)
            if torch.cuda.is_available():
                zx = zx.cuda()
            zx = Variable(zx)
            gen = VAE_net._decode(zx).cpu().view(batch_size, 3, 64, 64)

            save_image(gen.data, 'img_result_vae/vae_reconst_%d.png' % (epoch + 1), nrow=10)
            
    torch.save(VAE_net.state_dict(), 'saved_model_vae/vae.pth')