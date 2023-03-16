import torch
import numpy as np
from dataset import create_dataset
from unet3d import unet
from gan3d import Discriminator
import nibabel as nib
import os
from torchsummary import summary

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
        
        optimizer_G.zero_grad()
        generated_postop_data = gen_model(data)
        loss_G = generator_loss()

        concat_data = torch.cat([data, generated_postop_data], axis=1)

        print(concat_data.shape)
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