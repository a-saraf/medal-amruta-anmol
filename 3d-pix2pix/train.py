import torch
import numpy as np
from dataset import create_dataset

dir = '../DATA/00_Train/'

dataset = create_dataset(dir)

print(len(dataset))
for test_images in dataset:
    sample = test_images[0]
    print(sample.shape)
    print(sample)
    break

# 102 Images Dataloader
# dim 155 240 240 3D image