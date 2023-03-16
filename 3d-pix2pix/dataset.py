import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
        img = np.asarray(img)
        img = img.reshape(1, 240, 240, 155)
        if self.transform:
            img = self.transform(img)
        return img
    
def create_dataset(dir):
    dataset = BaseDataset(dir = dir, transform = transforms.Compose([transforms.ToTensor()]))
    dataloader = DataLoader(dataset)
    return dataloader
