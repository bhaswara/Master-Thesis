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

#Resnet WAE block

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

#Resnet WAE encoder

class ResNet34Enc(nn.Module):
    #for resnet18, num_blocks = [2,2,2,2]
    def __init__(self, num_Blocks=[3,4,6,3], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)


    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x

#resnet wae decoder
class ResNet34Dec(nn.Module):

    def __init__(self, num_Blocks=[3,4,6,3], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    
    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        return x

#Init model
class WAE(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder = ResNet34Enc(z_dim=z_dim)
        self.decoder = ResNet34Dec(z_dim=z_dim)

    def forward(self, x):
        z = self._encode(x)
        
        x_recon = self._decode(z)
        return x_recon, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, x):
        return self.decoder(x)


class Adversary(nn.Module):
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

WAE_net = WAE(z_dim=128)
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
            save_image(gen.data, 'img_result_resnetwae/wae_reconst_34128_%d.png' % (epoch + 1), nrow=10)
            

    torch.save(WAE_net.state_dict(), 'saved_model_resnetwae/WAE_34128.pth')
    torch.save(discriminator.state_dict(), 'saved_model_resnetwae/discriminator_34128.pth')
