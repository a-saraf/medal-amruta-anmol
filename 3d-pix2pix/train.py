import torch
import numpy as np
from dataset import create_dataset
from unet3d import unet
from gan3d import Discriminator
import nibabel as nib
import os
from torchsummary import summary
from torch.autograd import Variable

preop_dir = '../DATA/00_Train/'
postop_dir = '../DATA/01_Train/'

dir = {"pre":preop_dir, "post":postop_dir}

dataset = create_dataset(dir)
dataset_size = len(dataset)

gen_model = unet()
dis_model = Discriminator()

generator_loss = torch.nn.BCELoss()
discriminator_loss = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(gen_model.parameters())
optimizers_D = torch.optim.Adam(dis_model.parameters())

epochs = 100

for epoch in range(epochs):
    for i, data in enumerate(dataset):
        
        pre_data = data["pre"]
        post_data = data["post"]

        valid = Variable(torch.FloatTensor(1, 196).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(1, 196).fill_(0.0), requires_grad=False)

        optimizer_G.zero_grad()
        generated_postop_data = gen_model(pre_data)
        
        concat_data = torch.cat([pre_data, generated_postop_data], axis=1)

        loss_G = generator_loss(dis_model(concat_data), valid)
        loss_G.backward()
        optimizer_G.step()

        optimizers_D.zero_grad()

        real_loss = discriminator_loss(dis_model(torch.cat([pre_data, post_data], axis=1)), valid)
        fake_loss = discriminator_loss(dis_model(concat_data.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizers_D.step()

        print(generator_loss.item(), discriminator_loss.item())
        break
    break