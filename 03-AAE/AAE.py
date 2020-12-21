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


USE_CUDA = True
use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 100
n_epoch = 50
z_dim = 128

train_path = 'data_aligned_resize/train/'
writer = SummaryWriter()

cuda = True if torch.cuda.is_available() else False


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

class Encoder(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(Encoder, self).__init__()

        self.nc = nc
        self.z_dim = z_dim

        self.main = nn.Sequential(
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
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


    def forward(self, x):
        x = self.main(x)
        return x


class Decoder(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(Decoder, self).__init__()
        
        self.nc = nc
        self.z_dim = z_dim
        
        self.main = nn.Sequential(
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
            nn.ConvTranspose2d(128, nc, 1)
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        
        self.z_dim = z_dim
        
        self.main = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        dis = self.main(x)
        return dis

enc = Encoder(z_dim=128, nc=3).to(device)
dec = Decoder(z_dim=128, nc=3).to(device)
dis = Discriminator(z_dim=128).to(device)

adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.MSELoss()

optimizer_G = optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr = 3e-04, betas=(0.5, 0.999))

optimizer_disc = optim.Adam(dis.parameters(), lr = 1e-03, betas=(0.5, 0.999))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

enc.train()
dec.train()
dis.train()

for epoch in range(n_epoch):
    for batch_idx, (data,_) in enumerate(train_loader):

        valid = Variable(Tensor(data.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(data.shape[0], 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(data.type(Tensor))


        optimizer_G.zero_grad()

        encoded_imgs = enc(real_imgs)
        decoded_imgs = dec(encoded_imgs)

        adv_loss = F.binary_cross_entropy_with_logits(dis(encoded_imgs), valid)
        rec_loss = pixelwise_loss(decoded_imgs, real_imgs)

        enc_gan_loss = 0.001 * adv_loss + 0.999 * rec_loss

        #rec_loss.backward()
        enc_gan_loss.backward()
        optimizer_G.step()
    
        
        #dis  

        optimizer_disc.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (data.shape[0], z_dim))))

        dis_real_loss = F.binary_cross_entropy_with_logits(dis(z), valid)
        dis_fake_loss = F.binary_cross_entropy_with_logits(dis(encoded_imgs.detach()), fake)
        dis_loss = 0.5 * (dis_fake_loss + dis_real_loss)

        dis_loss.backward()
        optimizer_disc.step()
        
        
        print("Train Epoch: {} [{}/{} ({:.0f}%)] \tAAE_Loss: {:.6f}\tD_Loss: {:.6f}\tAdv_Loss: {:.6f}\tenc_gan_Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), rec_loss.item(), dis_loss.item(), adv_loss.item(), enc_gan_loss.item()))

        n_epochs = epoch*len(train_loader)+batch_idx
        writer.add_scalar('Discriminator_loss', dis_loss.data.cpu().numpy(), n_epochs)
        writer.add_scalar('aae_loss', rec_loss.data.cpu().numpy(), n_epochs)
        writer.add_scalar('Adversarial_loss', adv_loss.data.cpu().numpy(), n_epochs)
        writer.add_scalar('Enc_Gan_loss', enc_gan_loss.data.cpu().numpy(), n_epochs)
        
        if batch_idx % 5 == 0:

            zx = torch.randn(batch_size, z_dim)
            if torch.cuda.is_available():
                zx = zx.cuda()
            zx = Variable(zx)
            gen = dec(zx).cpu().view(batch_size, 3, 64, 64)
            save_image(gen.data, 'img_result_aae/aae_reconst_%d.png' % (epoch + 1), nrow=10)
            

    torch.save(enc.state_dict(), 'saved_model_aae/encoder.pth')
    torch.save(dec.state_dict(), 'saved_model_aae/decoder_.pth')
    torch.save(dis.state_dict(), 'saved_model_aae/discriminator_.pth')