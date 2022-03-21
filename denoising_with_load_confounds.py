import nibabel as nib
from nilearn import image as nimg
from nilearn.interfaces.fmriprep import load_confounds
from nilearn import masking
import os

'''
 Code in this file uses the function load_confounds for cleaning the data for one subject.
 load_confounds works with fmriprep directories and extracts confounds according to a specified strategy.
 Read more about it https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html#nilearn.interfaces.fmriprep.load_confounds
 Make sure that the nilearn package's version is 0.9.0 or newer
'''

# Subject's name, session and run should match the names in fmriprep output directory
sub = 'Y01'
ses = 1
run = 2

# ica aroma strategy "full" is the non-aggressive version
ica_aroma_strategy = "full"

# The relative path to the functional file in fmriprep directory
func_file = f'sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_run-{run}_' \
            f'space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz'

# First, we should choose the type of noise components to include strategy=["high_pass", "wm_csf", "ica_aroma"]
# Then, for each component, specify what type of confounds should be extracted with optional
# parameters: wm_csf="basic", ica_aroma=ica_aroma_strategy
confounds, mask = load_confounds(func_file, strategy=["high_pass", "wm_csf", "ica_aroma"],
                           wm_csf="basic", ica_aroma=ica_aroma_strategy)

func_img = nib.load(func_file)
# Check shape to make sure dimensions are ok after loading the image

# Save t_r of the functional image
t_r = 1.4

# Compute a matching brain mask
brain_mask = masking.compute_brain_mask(func_img)

# Call clean_img with the confounds dataframe the load_confounds returns
clean_img = nimg.clean_img(func_img, confounds=confounds, detrend=True, standardize=True,
                           t_r=t_r, mask_img=brain_mask)

# A path for saving the cleaned image (including the name of the file to save)
out_dir_file = os.path.join(f'load_confounds_dirs/sub-{sub}/ses-{ses}',
                            f'sub-{sub}_ses-{ses}_run-{run}_load_conf_strategy-{ica_aroma_strategy}.nii.gz')
# Save to out_dir_file
nib.save(clean_img, out_dir_file)

print("End")
