import glob
from PIL import Image
import numpy as np
import os

traditional_path = '../data/isic2018-rgb-299/'
segmentation_path = '../data/isic2018-seg-299/'
output_path = '../data/isic2018-onlyskin/'

###  Note that both traditional and segmentation images MUST have the same size. ###

if not os.path.exists(output_path):
    os.makedirs(output_path)

for file in glob.glob(segmentation_path + '*.png'):
	# Read the mask image.
	name = file.split('/')[-1]
	mask = Image.open(file)
	mask_np = np.array(mask)
	print(mask_np.shape)

    # Read the traditional image.
	traditional = Image.open(traditional_path + name)
	traditional_np = np.array(traditional)
	
	# Keep only the skin
	traditional_np[mask_np > 0] = 0

	new_image = Image.fromarray(traditional_np)
	new_image.save(output_path + name)
