import nibabel as nib
import numpy as np
import os

path_file = './BraTSReg_002_00_0000_flair.nii.gz'

t1 = nib.load(path_file)
t1 = t1.get_fdata()

print(t1.shape)

def make_dataset(dir):
    images = []
    
    for file_name in sorted(os.listdir(dir)):
        file_path = dir + '/' + file_name
        if(file_name.endswith('nii.gz')):
            images.append(file_name)
    return images
