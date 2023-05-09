# %%
import os, glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# %%

# STEP 1 - Load and visualize data
# First, we are going to indicate the path where our .nii files are located
# It is necessary to have 2 directories :
#       The 1st one contains the images we are going to use for training
#       The 2nd one contains the images we are going to use for testing     
# (So it is not necessary to have the same number of images in both)



dataInputPath = 'data/train/'
imagePathInput = os.path.join(dataInputPath, 'img/')
maskPathInput = os.path.join(dataInputPath, 'masks/')

dataOutputPath = 'data/slices/'
imageSliceOutput = os.path.join(dataOutputPath, 'img/img') 
# Exemple : it contains R1_im.nii
maskSliceOutput = os.path.join(dataOutputPath, 'masks/img') 
# Exemple : it contains R1_seg.nii

# Same, but for testing
dataTestInputPath = 'data/test/'
imageTestPathInput = os.path.join(dataTestInputPath, 'img/')
maskTestPathInput = os.path.join(dataTestInputPath, 'masks/')

dataTestOutputPath = 'data/slices/'
imageTestSliceOutput = os.path.join(dataTestOutputPath, 'imgTest/img')
 # Exemple : it contains T1_im.nii
maskTestSliceOutput = os.path.join(dataTestOutputPath, 'masksTest/img') 
# Exemple : it contains T1_seg.nii



def dir_creation(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

dir_creation(dataOutputPath)
dir_creation(imageSliceOutput)
dir_creation(maskSliceOutput)

dir_creation(dataTestOutputPath)
dir_creation(imageTestSliceOutput)
dir_creation(maskTestSliceOutput)


# STEP 2 - Image normalization
HOUNSFIELD_MIN = -1000
HOUNSFIELD_MAX = 2000
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

# STEP 3 - Slicing and saving
# Our .nii files have 3 axes, we can see it using 3DSlicer for example
# Here we decide the axes we want to use later for training and testing
# For example, if only SLICE_Z=True, then we will only have images from 
# the z-axe in "data/slices/img/" and on "data/slices/imgTest" 
  
SLICE_X = False
SLICE_Y = False
SLICE_Z = True  
SLICE_DECIMATE_IDENTIFIER = 3

# %% 

################## EXEMPLE WITH PLOT ##################
#######################################################

# Load image and see max min Hounsfield units
imgPath = os.path.join(imagePathInput, 'R1_im.nii')
img = nib.load(imgPath).get_fdata()
print(np.min(img), np.max(img), img.shape, type(img))

# Load mask and see max min Hounsfield units
maskPath = os.path.join(maskPathInput, 'R1_seg.nii')
mask = nib.load(maskPath).get_fdata()
print(np.min(mask), np.max(mask), mask.shape, type(mask))

# Show image slice in grayscale
imgSlice = img[:,:,600]
plt.imshow(imgSlice, cmap='gray')
plt.show()

# Show mask slice in grayscale
maskSlice = mask[:,:,600]
plt.imshow(maskSlice,cmap='gray')
plt.show()

# %% 
################## FUNCTION DEFINITIONS AND EXEMPLES ##################
#######################################################################

# Normalize image
def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

# Exemple
# nImg = normalizeImageIntensityRange(img)
# print(np.min(nImg), np.max(nImg), nImg.shape, type(nImg))

# Read image or mask volume
def readImageVolume(imgPath, normalize=False):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img
    
# Exemple    
# readImageVolume(imgPath, normalize=True)
# readImageVolume(maskPath, normalize=False)


# Save volume slice to file
def saveSlice(img, fname, path):
    img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')
    
# Exemple for saving one slice and one segmentation
# saveSlice(nImg[:,:,769], 'R1', imageSliceOutput)
# saveSlice(mask[:,:,769], 'R1', maskSliceOutput)


# Slice image in all directions and save with correct names and paths
def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    print(dimx, dimy, dimz)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSlice(vol[i,:,:], 
                      fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', 
                      path)
            
    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSlice(vol[:,i,:], 
                      fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y',
                      path)
            
    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSlice(vol[:,:,i], 
                      fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', 
                      path)
    return cnt

# %%
import time

# %% 
################## SAVE SLICES AND MASKS IN THE REPOSITORIES ##################
###############################################################################

t0 = time.perf_counter() 

# # Read and process image volumes for training
for index, filename in enumerate(sorted(glob.iglob(imagePathInput+'*.nii'))):
    img = readImageVolume(filename, True)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 'R'+str(index), imageSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')


# # Read and process image mask volumes for training
for index, filename in enumerate(sorted(glob.iglob(maskPathInput+'*.nii'))):
    img = readImageVolume(filename, False)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 'R'+str(index), maskSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')



# # Same, but with test data, so with images we are not going to use for training but for testing

for index, filename in enumerate(sorted(glob.iglob(imageTestPathInput+'*.nii'))):
    img = readImageVolume(filename, True)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 'T'+str(index), imageTestSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')


for index, filename in enumerate(sorted(glob.iglob(maskTestPathInput+'*.nii'))):
    img = readImageVolume(filename, False)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 'T'+str(index), maskTestSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')


timing_naive = time.perf_counter()-t0
print(
    f"Temps pour extraire toutes les images : {timing_naive:.4f} s."
)


# %% 
### EXEMPLE TO IDENTIFY WIDTH AND HEIGHT (used later on testing.py) ###
#######################################################################

filepath = "data/slices/img/img/R0-slice000_z.png"
img = Image.open(filepath)
  
# get width and height
width = img.width
height = img.height

print(width,height)
# %%
