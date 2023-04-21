import numpy as np
from PIL import Image
from sift_detector import SIFT
import cv2

import argparse
import pickle
import os
from os.path import isdir

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='SIFT')
	parser.add_argument('--input_path', type=str, default='./parrington/prtn10.jpg', help='Input image file path')
	args = parser.parse_args()

	img = np.asarray(Image.open(args.input_path).convert('L'), dtype = 'float32')
	# img = cv2.imread(args.input_path, 0).astype('float32') / 255
	print(img.shape)

	sift_detector = SIFT(img)
	_ = sift_detector.get_features()
	kp_pyr = sift_detector.kp_pyr

	# if not isdir('results'):
	# 	os.mkdir('results')

	# pickle.dump(sift_detector.kp_pyr, open('results/%s_kp_pyr.pkl' % args.output_prefix, 'wb'))
	# pickle.dump(sift_detector.feats, open('results/%s_feat_pyr.pkl' % args.output_prefix, 'wb'))

	# _, ax = plt.subplots(1, sift_detector.num_octave)
	
	# for i in range(sift_detector.num_octave):
	# 	ax[i].imshow(im)

	# 	scaled_kps = kp_pyr[i] * (2**i)
	# 	ax[i].scatter(scaled_kps[:,0], scaled_kps[:,1], c='r', s=2.5)

	# plt.show()