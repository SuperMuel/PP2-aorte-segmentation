import os, glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# STEP 1 - Load and visualize data
dataInputPath = 'data/train/'
imagePathInput = os.path.join(dataInputPath, 'img/')
maskPathInput = os.path.join(dataInputPath, 'masks/')

dataOutputPath = 'data/slices/'
imageSliceOutput = os.path.join(dataOutputPath, 'img/')
maskSliceOutput = os.path.join(dataOutputPath, 'masks/')

# STEP 2 - Image normalization
HOUNSFIELD_MIN = -1000
HOUNSFIELD_MAX = 2000
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

# STEP 3 - Slicing and saving
SLICE_X = True
SLICE_Y = True
SLICE_Z = False
SLICE_DECIMATE_IDENTIFIER = 3


# Load image and see max min Hounsfield units
imgPath = os.path.join(imagePathInput, 'R1_im.nii')
img = nib.load(imgPath).get_fdata()
print(np.min(img), np.max(img), img.shape, type(img))

# Load image mask and see max min Hounsfield units
maskPath = os.path.join(maskPathInput, 'R1_seg.nii')
mask = nib.load(maskPath).get_fdata()
print(np.min(mask), np.max(mask), mask.shape, type(mask))

# Show image slice
imgSlice = img[:,:,769]
plt.imshow(imgSlice, cmap='gray')
plt.show()

# Show mask slice 
maskSlice = mask[:,:,769]
plt.imshow(maskSlice,cmap='gray')
plt.show()

# Normalize image
def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

nImg = normalizeImageIntensityRange(img)
print(np.min(nImg), np.max(nImg), nImg.shape, type(nImg))

# Read image or mask volume
def readImageVolume(imgPath, normalize=False):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img
    
readImageVolume(imgPath, normalize=True)
readImageVolume(maskPath, normalize=False)


# Save volume slice to file
def saveSlice(img, fname, path):
    img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')
    
# saveSlice(nImg[:,:,769], 'R1', imageSliceOutput)
# saveSlice(mask[:,:,769], 'R1', maskSliceOutput)


# Slice image in all directions and save
def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    print(dimx, dimy, dimz)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSlice(vol[i,:,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)
            
    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSlice(vol[:,i,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)
            
    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSlice(vol[:,:,i], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)
    return cnt


# Convertir les img nii en png et les mettre dans le dossier 'data/slices/img'
# for index, filename in enumerate(sorted(glob.iglob(imagePathInput+'*.nii'))):
#     img = readImageVolume(filename, True)
#     print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
#     numOfSlices = sliceAndSaveVolumeImage(img, 'R'+str(index), imageSliceOutput)
#     print(f'\n{filename}, {numOfSlices} slices created \n')


# Convertir les masques nii en png et les mettre dans le dossier 'data/slices/masks'
# for index, filename in enumerate(sorted(glob.iglob(maskPathInput+'*.nii'))):
#     img = readImageVolume(filename, False)
#     print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
#     numOfSlices = sliceAndSaveVolumeImage(img, 'R'+str(index), maskSliceOutput)
#     print(f'\n{filename}, {numOfSlices} slices created \n')

filepath = "data/slices/img/img/R0-slice000_x.png"
img = Image.open(filepath)

print(img.width,img.height)
