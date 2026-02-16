import nibabel as nib

# "C:\Users\user\YongtaeC\macaque_fmri\caltech\site-caltech\sub-032183\ses-001\func\sub-032183_ses-001_task-movie_run-1_bold.nii.gz"

img = nib.load(r"C:\Users\user\YongtaeC\macaque_fmri\caltech\site-caltech\sub-032183\ses-001\func\sub-032183_ses-001_task-movie_run-1_bold.nii.gz")
data = img.get_fdata()
print("data shape: ", data.shape)
print("data type: ", data.dtype)
print("data min: ", data.min())
print("data max: ", data.max())
print("data mean: ", data.mean())
print("data std: ", data.std())
