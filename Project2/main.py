import cv2
import numpy as np
import argparse

from sift_detector import SIFT_get_features
from feature_matching import kd_tree_matching
from image_matching import RANSAC
from stitch import stitching
from drawplot import plot_matches


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='main function of Image Stitching Project')
	parser.add_argument('--draw_process', default=0, type=int, help='Set this to 1 is you want to see the process')
	args = parser.parse_args()
	darw = args.draw_process
	# image_path = ['./parrington/prtn%02d.jpg' %i for i in range(17, -1, -1)]
	image_path = ['./input/img{}.jpg'.format(i) for i in range(0, 5)]
	if darw:
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
			keypints1, descriptors1 = SIFT_get_features(previous_img, draw=darw)
			keypints2, descriptors2 = SIFT_get_features(next_img, draw=darw)
		else:
			previous_img = next_img
			keypints1, descriptors1 = keypints2, descriptors2
			next_img = cv2.imread(image_path[idx], 0).astype('float32')
			keypints2, descriptors2 = SIFT_get_features(next_img, draw=darw)
		print('{} features are extracted in image with id={}'.format(descriptors1.shape[0], idx))
		print('{} features are extracted in image with id={}'.format(descriptors2.shape[0], idx+1))
		# 2. Feature matching
		print('Feature matching...')
		matched_pairs = kd_tree_matching(previous_img, keypints1, descriptors1, keypints2, descriptors2)
		if darw:
			total_img = np.concatenate((previous_img, next_img), axis=1)
			plot_matches(matched_pairs, total_img)
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

		