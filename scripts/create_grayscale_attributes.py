import glob
import numpy as np
from scipy import misc
import os

atri_path = '../data/isic2018-attr-299/'
mask_path = '../data/isic2018-seg-299/'
output_path = '../data/isic2018-grayscale-attributes/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

file_name_arr = [] # [ISIC_00000, ISIC_000001, ISIC_000003, ...]
for file in glob.glob(atri_path+'*.png'):
	temp = file.split('/')[-1].split('_')
	file_name = temp[0]+'_'+temp[1]
	if file_name not in file_name_arr:
		file_name_arr.append(file_name)

for family in sorted(file_name_arr):
	# Navigate through each attribute for each lesion:
	for i, file in enumerate(glob.glob(atri_path+family+'*.png')):
		# Read the attribute mask
		read_image = misc.imread(file, flatten=True)
		read_image[read_image > 0] = 255
		read_image = np.int8(read_image/255)

		if i == 0:
			mask = misc.imread(mask_path+family+'.png', flatten=True)
			base_image = np.ones(read_image.shape, dtype=int) # Healthy Skin is 1
			mask[mask > 0] = 255
			mask = np.int8(mask/255)
			base_image += mask # Common Lesion is 2

		type_file = file.split('/')[-1].split('_')[3]

		if type_file == 'pigment': # 3
			base_image += read_image
		elif type_file == 'negative': # 4
			base_image += read_image*2
		elif type_file.startswith('streaks'): # 5
			base_image += read_image*3
		elif type_file == 'milia': # 6
			base_image += read_image*4
		elif type_file.startswith('globules'): #7
			base_image += read_image*5
		else:
			print(type_file, '... ERROR: Invalid File Found!!!!')
	base_image[base_image > 7] = 2
	
	for i in range(8):
		base_image[base_image == i] = i*30
	stacked_img = np.stack((base_image,)*3, axis=-1)
	misc.toimage(stacked_img, cmin=0, cmax=255).save(output_path+family+'.png')

