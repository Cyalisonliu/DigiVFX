import numpy as np
from PIL import Image
import cv2

from sift_detector import SIFT
from feature_matching import kd_tree_matching
from image_matching import RANSAC
from stitch import stitching
from drawplot import plot_matches
import matplotlib.pyplot as plt

import argparse
import pickle
import os
from os.path import isdir

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='SIFT')
	parser.add_argument('--input_path', type=str, default='./parrington/prtn10.jpg', help='Input image file path')
	args = parser.parse_args()

	# image_path = ['./photos/DSC_{}.jpg'.format(i) for i in range(6136, 6143)]
	image_path = ['./parrington/prtn17.jpg', './parrington/prtn16.jpg', './parrington/prtn15.jpg', './parrington/prtn14.jpg', './parrington/prtn13.jpg', './parrington/prtn12.jpg',
	       './parrington/prtn11.jpg', './parrington/prtn10.jpg', './parrington/prtn09.jpg', './parrington/prtn08.jpg', './parrington/prtn07.jpg', './parrington/prtn06.jpg',
		   './parrington/prtn05.jpg', './parrington/prtn04.jpg', './parrington/prtn03.jpg', './parrington/prtn02.jpg', './parrington/prtn01.jpg', './parrington/prtn00.jpg']
	offsets = [[0,0]]
	previous_img = None
	next_img = None

	for idx in range(1, len(image_path)):
		# 1. Apply SIFT to do feature detection
		print('[SIFT} Detecting features...')
		if idx == 1:
			previous_img = np.asarray(Image.open(image_path[idx-1]).convert('L'), dtype = 'float32')
			next_img = np.asarray(Image.open(image_path[idx]).convert('L'), dtype = 'float32')
			sift_detector1 = SIFT(previous_img)
			sift_detector2 = SIFT(next_img)
			keypints1, descriptors1 = sift_detector1.get_features()
			keypints2, descriptors2 = sift_detector2.get_features()
		else:
			previous_img = next_img
			keypints1, descriptors1 = keypints2, descriptors2
			next_img = np.asarray(Image.open(image_path[idx]).convert('L'), dtype = 'float32')
			sift_detector2 = SIFT(next_img)
			keypints2, descriptors2 = sift_detector2.get_features()
		print('{} features are extracted in image with id={}'.format(descriptors1.shape[0], idx))
		print('{} features are extracted in image with id={}'.format(descriptors2.shape[0], idx+1))
		# 2. Feature matching
		print('Feature matching...')
		matched_pairs = kd_tree_matching(previous_img, next_img, keypints1, descriptors1, keypints2, descriptors2)
		# total_img = np.concatenate((img1, img2), axis=1)
		# plot_matches(matched_pairs, total_img)
		print('Already got matched keypoints!')
		# 3. Apply RANSAC to find best offsets
		print('[RANSAC] Finding best offsets for these pair of images...')
		best_offset = RANSAC(matched_pairs[:, 1], matched_pairs[:, 0])
		offsets.append(best_offset)
		print('Already got best offsets!')

	# 4. stitch all images
	print('Processing image stitching...')
	result_image = stitching(image_path, offsets)
	result_image.save('result.jpg')
		