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

preop_dataset = create_dataset(preop_dir)
postop_dataset = create_dataset(postop_dir)

dataset_size = len(preop_dataset)

gen_model = unet()
dis_model = Discriminator()

generator_loss = torch.nn.BCELoss()
discriminator_loss = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(gen_model.parameters())
optimizers_D = torch.optim.Adam(dis_model.parameters())

epochs = 100

for epoch in range(epochs):
    for i, data in enumerate(preop_dataset):
        
        valid = Variable(torch.Tensor(196, 1).fill(1.0), requires_grad=False)
        fake = Variable(torch.Tensor(196, 1).fill(0.0), requires_grad=False)

        optimizer_G.zero_grad()
        generated_postop_data = gen_model(data)
        
        concat_data = torch.cat([data, generated_postop_data], axis=1)

        loss_G = generator_loss(dis_model(concat_data), valid)
        loss_G.backward()
        optimizer_G.step()

        optimizers_D.zero_grad()

        real_loss = discriminator_loss(dis_model(torch.cat([data, postop_dataset[i]], axis=1)), valid)
        fake_loss = discriminator_loss(dis_model(concat_data.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizers_D.step()

        print(generator_loss.item(), discriminator_loss.item())
        break
    break


# for file_name in sorted(os.listdir(dir)):
#     file_path = dir + '/' + file_name
#     if(file_name.endswith('nii.gz')):
#         img = nib.load(file_path)
#         img = img.get_fdata()
#         print(img.shape)
#     break

print(summary(model, (1, 155, 240, 240)))

# print(summary(model, (1, 155, 240, 240)))

# print(summary(model, (155, 240, 240)))

# for (idx, test_images) in enumerate(dataset):
#     sample = test_images
#     temp = model(sample)
#     print(temp.shape)
#     print(temp)
#     break

# 102 Images Dataloader
# dim 155 240 240 3D image