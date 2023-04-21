import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from scipy.interpolate import interp2d
import cv2

from utilis import generate_pyramid, get_keypoints, assign_orientation
from drawplot import draw_image, draw_keypoint


class SIFT(object):
    def __init__(self, img, s=3, num_octave=4, sigma=1.6, curvature_threshold=10, contrast_threshold=2.0, w=16):
        # Blur the image with a standard deviation of 0.5
        # Upsample the image by a factor of 2 using linear interpolation
        draw_image(img,'Original image')
        antialias_sigma = 0.5
        if antialias_sigma == 0:
            signal = img
        else:
            signal = gaussian_filter(img, sigma=antialias_sigma, mode='constant')
        # draw_image(signal,'Blur first image')
        # test = gaussian_filter(img, sigma=3.2)
        # draw_image(test, 'sigma=3.2')
        # Upsample the image by a factor of 2 using linear interpolation
        h, w = signal.shape
        signal = cv2.resize(signal, (int(w*2), int(h*2)), interpolation=cv2.INTER_NEAREST)
        # draw_image(signal, 'Blur 2x')
        self.subsample = [0.5]
        self.img = img
        self.signal = signal
        self.s = s
        self.sigma = sigma
        self.num_octave = num_octave
        self.contrast_threshold = contrast_threshold
        self.curvature_threshold = (curvature_threshold+1)**2 / curvature_threshold
        self.w = w

    def get_features(self):
        """
            1. Generate gaussain pyrimid and DoG pyrimid
        """
        gaussian_pyr, DOG_pyr, subsample = generate_pyramid(self.signal, self.num_octave, self.s, self.sigma, self.subsample)
        # for i, gaussian_per in enumerate(gaussian_pyr):
        #     draw_image(gaussian_per[0], f'Gaussain image ${i+1}')
        # for i, dog_per in enumerate(DOG_pyr):
        #     draw_image(dog_per[0], f'DOG image ${i+1}')

        """
            2. Detect keypoints over DoG pyrimid
        """
        kp_pyr, raw_keypoints, contrast_keypoints, curve_keypoints = get_keypoints(DOG_pyr, self.num_octave, self.s, subsample, self.contrast_threshold, self.curvature_threshold)
        draw_keypoint(self.img, raw_keypoints, contrast_keypoints, curve_keypoints)
        # print(len(kp_pyr))
        # print(len(kp_pyr[0]))
        # print(len(kp_pyr[1]))
        """
            3. Assign orientations to the keypoints
        """
        feats = []
        kp_pyr = assign_orientation(kp_pyr, gaussian_pyr, self.s, self.num_octave, subsample)
        # for i, DoG_octave in enumerate(DOG_pyr):
        #     kp_pyr[i] = assign_orientation(kp_pyr[i], DoG_octave)
            # feats.append(get_local_descriptors(kp_pyr[i], DoG_octave))

        # self.kp_pyr = kp_pyr
        # self.feats = feats

        # return feats