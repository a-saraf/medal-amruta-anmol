import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import os
import numpy as np

path_file = '../DATA/00_Train/'

def make_dataset(dir):
    images = []
    
    for file_name in sorted(os.listdir(dir)):
        file_path = dir + '/' + file_name
        if(file_name.endswith('nii.gz')):
            images.append(file_path)
    return images

class BaseDataset(Dataset):
    def __init__(self, dir, transform = None):
        self.dir = dir
        self.images = make_dataset(dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = nib.load(self.images[index])
        img = img.get_fdata()
        if self.transform:
            img = self.transform(img)
        return img
    
def create_dataset(dir):
    dataset = BaseDataset(dir, transform=transforms.Compose([ToTensor(expand_dims=True)]))
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader

class ToTensor:
    def __init__(self, expand_dims, dtype=np.float32):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D or 4D images'
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)
        return torch.from_numpy(m.astype(dtype=self.dtype))