import nibabel as nib
import numpy as np

path_file = './BraTSReg_002_00_0000_flair.nii.gz'

t1 = nib.load(path_file)
t1 = t1.get_fdata()

print(t1.shape)