import numpy as np
from PIL import Image
import cv2

from sift_detector import SIFT
from feature_matching import kd_tree_matching
from image_matching import RANSAC
from stitch import stitching
from drawplot import plot_matches
import matplotlib.pyplot as plt


if __name__ == '__main__':
	# image_path = ['./parrington/prtn%02d.jpg' %i for i in range(17, -1, -1)]
	image_path = ['./input_best/img{}.jpg'.format(i) for i in range(0, 5)]
	print(image_path)
	offsets = [[0,0]]
	previous_img = None
	next_img = None

	for idx in range(1, len(image_path)):
		# 1. Apply SIFT to do feature detection
		print('[SIFT} Detecting features...')
		if idx == 1:
			previous_img = cv2.imread(image_path[idx-1], 0).astype('float32')
			next_img = cv2.imread(image_path[idx], 0).astype('float32')
			sift_detector1 = SIFT(previous_img)
			sift_detector2 = SIFT(next_img)
			keypints1, descriptors1 = sift_detector1.get_features()
			keypints2, descriptors2 = sift_detector2.get_features()
		else:
			previous_img = next_img
			keypints1, descriptors1 = keypints2, descriptors2
			next_img = cv2.imread(image_path[idx], 0).astype('float32')
			sift_detector2 = SIFT(next_img)
			keypints2, descriptors2 = sift_detector2.get_features()
		print('{} features are extracted in image with id={}'.format(descriptors1.shape[0], idx))
		print('{} features are extracted in image with id={}'.format(descriptors2.shape[0], idx+1))
		# 2. Feature matching
		print('Feature matching...')
		matched_pairs = kd_tree_matching(previous_img, next_img, keypints1, descriptors1, keypints2, descriptors2)
		# total_img = np.concatenate((previous_img, next_img), axis=1)
		# plot_matches(matched_pairs, total_img)
		print('Already got matched keypoints!')
		# 3. Apply RANSAC to find best offsets
		print('[RANSAC] Finding best offsets for these pair of images...')
		best_offset = RANSAC(matched_pairs[:, 1], matched_pairs[:, 0])
		offsets.append(best_offset)
		print('Already got best offsets!', offsets)

	# 4. stitch all images
	print('Processing image stitching...')
	result_image, crop_img = stitching(image_path, offsets)
	cv2.imwrite('result_noCrop.jpg', result_image)
	cv2.imwrite('result_crop.jpg', crop_img)

		