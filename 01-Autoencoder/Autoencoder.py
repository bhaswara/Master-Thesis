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
z_dim = 128
train_path = 'data_aligned_resize/train/'
test_path = 'data_aligned_resize/val/'
writer = SummaryWriter()


USE_CUDA = True
use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#To Flatten the shape
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

#Load training data
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

#Load testing data
test_dataset = ImageFolder(
        root=test_path,
        transform=torchvision.transforms.ToTensor()
    )
test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )


#Initialize the weight
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


#Initialize AE model and optimizer
class AE(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(AE, self).__init__()
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


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


AE_net = AE(z_dim=128, nc=3)

if torch.cuda.is_available():
    AE_net = AE_net.cuda()

optim_AE = optim.Adam(AE_net.parameters(), lr = 1e-03, betas=(0.5, 0.999))
loss_fn = nn.MSELoss()

#Train

AE_net.train()

for epoch in range(n_epoch):
    for batch_idx, datax in enumerate(train_loader, 0):
        
        data,_ = datax

        if torch.cuda.is_available():
            data = data.cuda()
        
        data = Variable(data)
        
        recons_x, code = AE_net(data)

        optim_AE.zero_grad()
        loss_ae = loss_fn(recons_x, data)
        loss_ae.backward()
        optim_AE.step()
        
        
        print("Train Epoch: {} [{}/{} ({:.0f}%)] \tAE_Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_ae.item()))

        n_epochs = epoch*len(train_loader)+batch_idx
        writer.add_scalar('ae_loss', loss_ae.data.cpu().numpy(), n_epochs)
        
        #Show the image result
        if batch_idx % 5 == 0:
            test_iter = iter(test_loader)
            test_data = next(test_iter)

            code = AE_net._encode(Variable(test_data[0]).cuda())

            gen = AE_net._decode(code).cpu().view(batch_size, 3, 64, 64)
            save_image(gen.data, 'img_result_ae/ae_reconst_%d.png' % (epoch + 1), nrow=10)
            

    #Save the weight
    torch.save(AE_net.state_dict(), 'saved_model_ae/AE.pth')