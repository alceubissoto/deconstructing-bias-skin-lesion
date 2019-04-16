import glob
from PIL import Image
import numpy as np
import os

traditional_path = '../data/isic2018-rgb-299/'
segmentation_path = '../data/isic2018-seg-299/'
output_path = '../data/isic2018-bbox70/'

###  Note that both traditional and segmentation images are resized to be 299x299. ###

if not os.path.exists(output_path):
    os.makedirs(output_path)

for file in glob.glob(segmentation_path + '*.png'):
	name = file.split('/')[-1]
	image = Image.open(file)
	image_np = np.array(image)
	print(image_np.shape)

	top = 100000
	bottom = 0
	left = 100000
	right = 0
	# Find the bounding box coordinates.
	for i in range(len(image_np)):
		for j in range(len(image_np[0])):
			pix = image_np[i, j]
			if pix == 255:
				if i < top:
					top = i
				if i > bottom:
					bottom = i
				if j < left:
					left = j
				if j > right:
					right = j

	middle_x = (right + left)/2
	middle_y = (top + bottom)/2

    # If the traditional bounding box occupy less than 70% of the image, place a 250x250 square in the middle of the lesion.
    # Else, return the traditional bounding box.
	new_image = Image.open(traditional_path + name)
	new_image = np.array(new_image)
	if (right - left) * (bottom - top) < 0.7 * 299 * 299:
		for i in range(len(image_np)):
			for j in range(len(image_np[0])):
				if i > middle_y - 125 and i < middle_y + 125 and j > middle_x - 125 and j < middle_x + 125:
					new_image[i, j] = 0
		new_image = Image.fromarray(new_image)
		new_image.save(output_path + name)
	else:
		for i in range(len(image_np)):
			for j in range(len(image_np[0])):
				if i > top and i < bottom and j > left  and j < right:
					new_image[i, j] = 0
		new_image = Image.fromarray(new_image)
		new_image.save(output_path + name)
