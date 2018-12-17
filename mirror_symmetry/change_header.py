import os
import nibabel as nib
import numpy as np

img_path = os.path.join(os.getcwd(), os.pardir, os.pardir,
                        'symmetry_mask.nii.gz')
img_s_path = os.path.join(os.getcwd(), os.pardir, os.pardir,
                          'symmetry_mask_2.nii.gz')

img_nii = nib.load(img_path)
img_data = img_nii.get_fdata()
img_affine = img_nii.get_qform()

print(img_affine)

tmp = np.copy(img_affine[:, 0])
img_affine[:, 0] = img_affine[:, 1]
img_affine[:, 1] = tmp

print(img_affine)

img_nii = nib.Nifti1Image(img_data, img_affine)
nib.save(img_nii, img_s_path)
