import numpy as np
from nilearn import input_data
from nilearn import image as nimg
from nilearn import plotting as nplot
import os

sub = 'Y01'
ses = 1
run = 2
t_r = 1.4

# The relative path to the functional file in fmriprep directory
func_file = f'Clean Data/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_run-{run}_load_conf_strategy-full.nii.gz'

# Coordinates of the seed sphere center
coords = [(-42, -16, 52)]

# Extract the time series from the functional imaging within the sphere.
# The sphere is centered at coords and will have the radius we pass the NiftiSpheresMasker function (here 8 mm).
clean_img = nimg.load_img(func_file)

seed_masker = input_data.NiftiSpheresMasker(coords, radius=8, detrend=True, standardize=True, t_r=t_r,
                                            memory='nilearn_cache', memory_level=1, verbose=0)

# extract the mean time series
seed_time_series = seed_masker.fit_transform(clean_img)

# brain-wide voxel-wise time series, using nilearn.input_data.NiftiMasker with the same input arguments as in the
# seed_masker in addition to smoothing with a 6 mm kernel
brain_masker = input_data.NiftiMasker(smoothing_fwhm=6, detrend=True, standardize=True,
                                      t_r=1.4, memory='nilearn_cache', memory_level=1, verbose=0)

# extract the brain-wide voxel-wise time series
brain_time_series = brain_masker.fit_transform(clean_img)

# a reminder that np.dot is matrix multiplication when arguments are 2d arrays
seed_to_voxel_correlations = np.dot(brain_time_series.T, seed_time_series) / \
                             seed_time_series.shape[0]

seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)

# Change working directory to save files
code_dir = os.getcwd()
out_dir = f'Seed_corr_dirs'
os.chdir(out_dir)

# Save the 3d image as a nii.gz file
seed_to_voxel_correlations_img.to_filename(f'seed_corr_sub-{sub}_ses-{ses}_run{run}.nii.gz')

# Save an interactive view that can viewed from a web browser
interactive_img = nplot.view_img(seed_to_voxel_correlations_img, threshold=0.25,
                                 cut_coords=coords[0])  # html interactive view
interactive_img.save_as_html(f'seed_corr_sub-{sub}_ses-{ses}_run{run}.html')

os.chdir(code_dir)
