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
file_log = open("train_log.txt","a")

dir = {"pre":preop_dir, "post":postop_dir}

dataset = create_dataset(dir)
dataset_size = len(dataset)

gen_model = unet()
dis_model = Discriminator()

generator_loss = torch.nn.BCELoss()
discriminator_loss = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(gen_model.parameters())
optimizers_D = torch.optim.Adam(dis_model.parameters())

epochs = 500

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
        loss_D = (real_loss + fake_loss) / 2

        loss_D.backward()
        optimizers_D.step()

        print("epoch", epoch, "loss_G", loss_G.item(), "loss_D", loss_D.item())
        file_log.write("epoch", epoch, "loss_G", loss_G.item(), "loss_D", loss_D.item())

        if((i+1)%50 == 0):
            path = '../ckpt_models/gen_models/gen_model_epoch' + str(i+1) + 'pth'
            torch.save(gen_model, path)
            path = '../ckpt_models/dis_models/dis_model_epoch' + str(i+1) + 'pth'
            torch.save(dis_model, path)

file_log.close()