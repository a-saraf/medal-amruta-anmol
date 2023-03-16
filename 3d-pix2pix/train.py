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

cuda = True if torch.cuda.is_available() else False

dataset = create_dataset(dir)
dataset_size = len(dataset)

gen_model = unet()
dis_model = Discriminator()

generator_loss = torch.nn.BCELoss()
discriminator_loss = torch.nn.BCELoss()

if cuda:
    gen_model.cuda()
    dis_model.cuda()
    generator_loss.cuda()
    discriminator_loss.cuda()

optimizer_G = torch.optim.Adam(gen_model.parameters())
optimizers_D = torch.optim.Adam(dis_model.parameters())

epochs = 500

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(epochs):
    for i, data in enumerate(dataset):
        
        pre_data = data["pre"]
        post_data = data["post"]

        valid = Variable(Tensor(1, 196).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(1, 196).fill_(0.0), requires_grad=False)

        pre_data = Variable(pre_data.type(Tensor))
        post_data = Variable(post_data.type(Tensor))

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

    file_log = open("train_log.txt","a")
    file_log.write("epoch," + str(epoch) + "loss_G,"+ str(loss_G.item()) + "loss_D," + str(loss_D.item()) + '\n')
    file_log.close()

    if((epoch+1)%50 == 0):
        path = '../ckpt_models/gen_models/gen_model_epoch' + str(epoch + 1) + 'pth'
        torch.save(gen_model, path)
        path = '../ckpt_models/dis_models/dis_model_epoch' + str(epoch + 1) + 'pth'
        torch.save(dis_model, path)

