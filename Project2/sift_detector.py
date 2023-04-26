import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from scipy.interpolate import interp2d

from utilis import generate_pyramid, get_keypoints, assign_orientation, generate_descriptor
from drawplot import draw_image, draw_keypoint


class SIFT(object):
    def __init__(self, img, s=3, num_octave=3, sigma=1.8, curvature_threshold=7.0, contrast_threshold=4.0):
        # Blur the image with a standard deviation of 0.5
        # Upsample the image by a factor of 2 using linear interpolation
        # draw_image(img,'Original image')
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
        # draw_keypoint(self.img, raw_keypoints, contrast_keypoints, curve_keypoints)

        """
            3. Assign orientations to the keypoints and get descriptor
        """
        kp_pyr, orient, scale = assign_orientation(kp_pyr, gaussian_pyr, self.s, self.num_octave, subsample)
        kp_pyr, descriptor = generate_descriptor(kp_pyr, gaussian_pyr, orient, scale, subsample)
        # print(kp_pyr.shape, descriptor.shape)
        # draw_keypoint(self.img, kp_pyr)

        return kp_pyr, descriptor