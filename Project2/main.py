import numpy as np
from PIL import Image

from sift_detector import SIFT
from feature_matching import kd_tree_matching
from generate_panorama import RANSAC
from drawplot import plot_matches

import argparse
import pickle
import os
from os.path import isdir

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='SIFT')
	parser.add_argument('--input_path', type=str, default='./parrington/prtn10.jpg', help='Input image file path')
	args = parser.parse_args()

	# img = np.asarray(Image.open(args.input_path).convert('L'), dtype = 'float32')
	# # img = cv2.imread(args.input_path, 0).astype('float32') / 255
	# print(img.shape)

	# sift_detector = SIFT(img)
	# _ = sift_detector.get_features()
	# kp_pyr = sift_detector.kp_pyr

	image_set_size = ['./parrington/prtn13.jpg', './parrington/prtn12.jpg']

	print('[SIFT} Detecting features...')
	img1 = np.asarray(Image.open(image_set_size[0]).convert('L'), dtype = 'float32')
	img2 = np.asarray(Image.open(image_set_size[1]).convert('L'), dtype = 'float32')
	sift_detector1 = SIFT(img1)
	sift_detector2 = SIFT(img2)
	keypints1, descriptors1 = sift_detector1.get_features()
	keypints2, descriptors2 = sift_detector2.get_features()
	print('{} features are extracted in first image'.format(descriptors1.shape[0]))
	print('{} features are extracted in second image'.format(descriptors2.shape[0]))
   
	print('Feature matching...')
	matched_pairs = kd_tree_matching(img1, img2, keypints1, descriptors1, keypints2, descriptors2)
	total_img = np.concatenate((img1, img2), axis=1)
	plot_matches(matched_pairs, total_img)
	print('Already got matched keypoints!')

	print('[RANSAC] Finding best homography H for these pair of images...')
	RANSAC(matched_pairs[:, 0], matched_pairs[:, 1])
    #  print ' | Find best shift using RANSAC .... '; sys.stdout.flush()
    #     shift = findshift.RANSAC(matched_pairs)
    #     shifts.append(shift)
    #     print ' | | best shift ', shift
    # print 'Completed feature matching! Here are all shifts:'; sys.stdout.flush()
    # print shifts;  sys.stdout.flush()