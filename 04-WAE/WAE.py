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

import math

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
z_var = 2

global_epoch = 0
global_iter = 0


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


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


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class WAE(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(WAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*4*4)),                                 
            nn.Linear(1024*4*4, z_dim)                            
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*8*8),                           
            View((-1, 1024, 8, 8)),                              
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Adversary(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim):
        super(Adversary, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),                                
            nn.ReLU(True),
            nn.Linear(512, 512),                                  
            nn.ReLU(True),
            nn.Linear(512, 512),                                  
            nn.ReLU(True),
            nn.Linear(512, 512),                                  
            nn.ReLU(True),
            nn.Linear(512, 1),                                    
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)

def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def gan_loss(sample_qz, sample_pz):
    logits_qz = discriminator(sample_qz)
    logits_pz = discriminator(sample_pz)

    loss_qz = F.binary_cross_entropy_with_logits(logits_qz, torch.zeros_like(logits_qz))

    #loss_pz = d_loss_function(logits_pz, torch.ones_like(logits_pz))
    loss_pz = F.binary_cross_entropy_with_logits(logits_pz, torch.ones_like(logits_pz))

    #loss_qz_trick = d_loss_function(logits_qz, torch.ones_like(logits_qz))
    loss_qz_trick = F.binary_cross_entropy_with_logits(logits_qz, torch.ones_like(logits_qz))

    loss_adversary = lam * (loss_qz + loss_pz)
    loss_penalty = loss_qz_trick
    return (loss_adversary, logits_qz, logits_pz), loss_penalty

def sample_pz(batch_size, z_size=128):
    return Variable(torch.normal(torch.zeros(batch_size, z_size), std=1).cuda())

#Call WAE model
WAE_net = WAE(z_dim=128, nc=3)
discriminator = Adversary(z_dim=128)

if torch.cuda.is_available():
    WAE_net, discriminator = WAE_net.cuda(), discriminator.cuda()

optim_WAE = optim.Adam(WAE_net.parameters(), lr = 3*1e-04, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr = 1e-03, betas=(0.5, 0.999))

WAE_net.train()

for epoch in range(n_epoch):
    for batch_idx, datax in enumerate(train_loader, 0):
        
        data,_ = datax

        if torch.cuda.is_available():
            data = data.cuda()
        
        data = Variable(data)
        sample_noise = sample_pz(batch_size, z_dim)

        recon_x, encoded = WAE_net(data)

        decoded = WAE_net._decode(sample_noise)

        recon_loss = F.mse_loss(recon_x, data, size_average=False).div(batch_size)
        loss_gan, loss_penalty = gan_loss(encoded, sample_noise)
        loss_wae = recon_loss + lam * loss_penalty
        loss_adv = loss_gan[0]
    
        optim_WAE.zero_grad()
        loss_wae.backward(retain_graph=True)
        optim_WAE.step()

        discriminator.zero_grad()
        loss_adv.backward()
        optim_D.step()
        
        
        print("Train Epoch: {} [{}/{} ({:.0f}%)] \tWAE_Loss: {:.6f}\tD_Loss: {:.6f}\tQ_Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_wae.item(), loss_adv.item(), loss_penalty.item()))

        n_epochs = epoch*len(train_loader)+batch_idx
        writer.add_scalar('Discriminator_loss', loss_adv.data.cpu().numpy(), n_epochs)
        writer.add_scalar('Wae_loss', loss_wae.data.cpu().numpy(), n_epochs)
        writer.add_scalar('Adversarial_loss', loss_penalty.data.cpu().numpy(), n_epochs)
        
        if batch_idx % 5 == 0:

            zx = torch.randn(batch_size, z_dim)
            if torch.cuda.is_available():
                zx = zx.cuda()
            zx = Variable(zx)
            gen = WAE_net._decode(zx).cpu().view(batch_size, 3, 64, 64)
            save_image(gen.data, 'img_result_wae/wae_reconst_%d.png' % (epoch + 1), nrow=10)
            

    torch.save(WAE_net.state_dict(), 'saved_model_wae/WAE.pth')
    torch.save(discriminator.state_dict(), 'saved_model_wae/discriminator_.pth')