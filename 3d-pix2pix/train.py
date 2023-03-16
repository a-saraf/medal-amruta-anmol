import torch
import numpy as np
from dataset import create_dataset
from unet3d import UnetGenerator
from gan3d import Discriminator
import nibabel as nib
import os
from torchsummary import summary

model = Discriminator()

dir = '../DATA/00_Train/'

# for file_name in sorted(os.listdir(dir)):
#     file_path = dir + '/' + file_name
#     if(file_name.endswith('nii.gz')):
#         img = nib.load(file_path)
#         img = img.get_fdata()
#         print(img.shape)
#     break

dataset = create_dataset(dir)

print(summary(model, (1, 155, 240, 240)))

# print(summary(model, (155, 240, 240)))

# for test_images in dataset:
#     sample = test_images[0]
#     temp = model.forward(sample)
#     print(temp)
#     break

# 102 Images Dataloader
# dim 155 240 240 3D image