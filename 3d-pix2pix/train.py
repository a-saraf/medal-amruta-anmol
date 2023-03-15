import torch
import numpy as np
from dataset import create_dataset

dir = '../DATA/00_Train/'

dataset = create_dataset(dir)

print(len(dataset))
