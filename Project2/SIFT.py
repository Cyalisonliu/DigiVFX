import numpy as np
import math
import cv2
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt


def DoGDetector(im, octaves=4, intervals=2, object_mask=None, contrast_threshold=0.03,
                curvature_threshold=10.0):
    """
    Detects keypoints in the given image using the Difference of Gaussian (DoG) method.

    Args:
        im (ndarray): The input image, with pixel values normalize to lie between [0,1].
        octaves (int): The number of octaves to search for keypoints (default=4).
        intervals (int): The number of geometrically sampled intervals to divide
            each octave into when searching for keypoints (default=2).
        object_mask (ndarray): A binary mask specifying the location of the object in
            the image to search for keypoints on.  If not specified, the whole
            image is searched.
        contrast_threshold (float): The threshold on the contrast of the DoG extrema
            before classifying them as keypoints (default=0.03).
        curvature_threshold (float): The upper bound on the ratio between the principal
            curvatures of the DoG extrema before classifying it as a keypoint
            (default=10.0).

    Returns:
        pos (ndarray): An Nx2 matrix containing the (x,y) coordinates of the keypoints
            stored in rows.
        scale (ndarray): An Nx3 matrix with rows describing the scale of each keypoint (i.e.,
            first column specifies the octave, second column specifies the interval, and
            third column specifies sigma).
        orient (ndarray): A Nx1 vector containing the orientations of the keypoints [-pi,pi).
        desc (ndarray): An Nx128 matrix with rows containing the feature descriptors
            corresponding to the keypoints.
    """
    if object_mask == None:
        object_mask = np.ones_like(im)
   
    # Implementation goes here

    """
        Blur the image with a standard deviation of 0.5 to prevent aliasing
        and then upsample the image by a factor of 2 using linear interpolation.
        Lowe claims that this increases the number of stable keypoints by 
        a factor of 4.
    """

    print("Doubling image size for first octave...")

    # Blur the image with a standard deviation of 0.5
    antialias_sigma = 0.5
    g = cv2.getGaussianKernel(ksize=5, sigma=antialias_sigma)
    signal = cv2.filter2D(im, -1, g)
    # Upsample the image by a factor of 2 using linear interpolation
    h, w = signal.shape
    signal = cv2.resize(signal, int(w*2), int(h*2), interpolation=cv2.INTER_LINEAR)
    subsample = [0.5]  # subsampling rate for doubled image is 1/2

    """
        The next step of the algorithm is to generate the gaussian and difference-of-
        gaussian (DOG) pyramids.  These pyramids will be stored as two cell arrays,
        gauss_pyr{orient,interval} and DOG_pyr{orient,interval}, respectively.  In order
        to detect keypoints on s intervals per octave, we must generate s+3 blurred
        images in the gaussian pyramid.  This is becuase s+3 blurred images generates
        s+2 DOG images, and two images are needed (one at the highest and one lowest scales 
        of the octave) for extrema detection.
    """

    gauss_per_pyr = []
    gauss_pyr = []

    """
        Generate the first image of the first octave of the gaussian pyramid
        by preblurring the doubled image with a gaussian with a standard deviation
        of 1.6.  This choice for sigma is a trade off between repeatability and
        efficiency.
    """

    print("Prebluring image...")

    # Pre-blur the image with a standard deviation of preblur_sigma
    preblur_sigma = np.sqrt(np.power(np.sqrt(2), 2) - np.power(2 * antialias_sigma, 2))
    if preblur_sigma == 0:
        gauss_per_pyr = [signal]
    else:
        g = cv2.getGaussianKernel(ksize=5, sigma=preblur_sigma)
        gauss_per_pyr = [cv2.filter2D(signal, -1, g)]

    # The initial blurring for the first image of the first octave of the pyramid.
    initial_sigma = np.sqrt( (2*antialias_sigma)**2 + preblur_sigma**2 )

    # Keep track of the absolute sigma for the octave and scale
    # absolute_sigma = np.zeros((octaves,intervals+3))
    # absolute_sigma[0,0] = initial_sigma * subsample[0]

    # Keep track of the filter sizes and standard deviations used to generate the pyramid
    # filter_size = np.zeros((octaves,intervals+3))
    # filter_sigma = np.zeros((octaves,intervals+3))

    # Generate the remaining levels of the geometrically sampled gaussian and DOG pyramids
    print("Expanding the Gaussian and DOG pyramids...\n")

    DOG_pyr = []
    for octave in range(octaves):
        print(f"\tProcessing octave {octave}: image size {gauss_pyr.shape[1]} x {gauss_pyr.shape[0]} subsample {subsample[octave]:.1f}")
        # print(f"\t\tInterval 1 sigma {absolute_sigma[octave, 1]}")
        
        sigma = initial_sigma
        g = cv2.getGaussianKernel(ksize=5, sigma=sigma)
        # filter_size[octave, 0] = len(g)
        # filter_sigma[octave, 0] = sigma
        DOG_per_pyr = []
        
        for interval in range(1, intervals + 3):
            # Compute the standard deviation of the gaussian filter needed to produce the 
            # next level of the geometrically sampled pyramid.
            k = 2**(1 / intervals)
            sigma_f = np.sqrt(k**2 - 1) * sigma
            g = gaussian_filter(sigma_f)
            sigma *= (2**(1 / intervals))
            
            # Keep track of the absolute sigma
            # absolute_sigma[octave, interval] = sigma * subsample[octave]
            
            # Store the size and standard deviation of the filter for later use
            # filter_size[octave, interval] = len(g)
            # filter_sigma[octave, interval] = sigma
            
            gauss_per_pyr.append(cv2.filter2D(gauss_per_pyr[interval - 1], -1, g))
            DOG_per_pyr.append(gauss_per_pyr[interval] - gauss_per_pyr[interval - 1])
            # if interactive >= 1:
            #     print(f"\t\tInterval {interval} sigma {absolute_sigma[octave, interval]}")
        
        DOG_pyr.append(DOG_per_pyr)
        gauss_pyr.append(gauss_per_pyr)
        gauss_per_pyr = []

        if octave < octaves - 1:
            # Subsample this image by a factor of 2 to procuduce the first image of the next octave.
            h, w = gauss_pyr[octave][-1].shape
            gauss_per_pyr.append(cv2.resize(gauss_pyr[octave][-1], int(w/2), int(h/2), interpolation=cv2.INTER_LINEAR))
    
            # absolute_sigma[octave + 1, 0] = absolute_sigma[octave, intervals + 1]
            subsample.append(subsample[-1] * 2)

    """
        The next step is to detect local maxima in the DOG pyramid.  When
        a maximum is found, two tests are applied before labeling it as a 
        keypoint.  First, it must have sufficient contrast.  Second, it should
        not be and edge point (i.e., the ratio of principal curvatures at the
        extremum should be below a threshold).
    """
    # Compute threshold for the ratio of principle curvature test applied to
    # the DOG extrema before classifying them as keypoints.
    curvature_threshold = ((curvature_threshold + 1)**2)/curvature_threshold

    # 2nd derivative kernels 
    xx = np.array([1, -2, 1])
    yy = xx[:, np.newaxis]
    xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4

    # Initialize pyramids to store extrema locations
    print("Locating keypoints...")
    extreme_pyr = []

    raw_keypoints = []
    contrast_keypoints = []
    curve_keypoints = []
    loc = []
    for octave in range(1, octaves+1):
        print("\tProcessing octave", octave)
        loc_per_octave = []
        for interval in range(2, intervals+2):
            contrast_mask = abs(DOG_pyr[octave][:,:,interval]) >= contrast_threshold
            loc_per_octave.append(np.zeros_like(DOG_pyr[octave][:,:,interval]))

            for y in range(1, DOG_pyr[octave].shape[0]-1):
                for x in range(1, DOG_pyr[octave].shape[1]-1):
                    # Only check for extrema where the object mask is 1
                    if object_mask[int(round(y*subsample[octave])), int(round(x*subsample[octave]))] == 1:
                        if contrast_mask[y,x] == 1:
                            # Check for a max or a min across space and scale
                            tmp = DOG_pyr[octave][y-1:y+2, x-1:x+2, interval-1:interval+2]
                            pt_val = tmp[1,1,1]
                            if (pt_val == np.min(tmp)) or (pt_val == np.max(tmp)):
                                # The point is a local extrema of the DOG image.
                                raw_keypoints = np.vstack((raw_keypoints, [x*subsample[octave], y*subsample[octave]]))

                                if abs(DOG_pyr[octave][y,x,interval]) >= contrast_threshold:
                                    # The DOG image at the extrema is above the contrast threshold.
                                    contrast_keypoints = np.vstack((contrast_keypoints, raw_keypoints[-1,:]))

                                    # Compute the entries of the Hessian matrix at the extrema location.
                                    Dxx = np.sum(DOG_pyr[octave][y,x-1:x+2,interval] * xx)
                                    Dyy = np.sum(DOG_pyr[octave][y-1:y+2,x,interval] * yy)
                                    Dxy = np.sum(np.sum(DOG_pyr[octave][y-1:y+2,x-1:x+2,interval] * xy))

                                    # Compute the trace and the determinant of the Hessian.
                                    Tr_H = Dxx + Dyy
                                    Det_H = Dxx*Dyy - Dxy**2

                                    # Compute the ratio of the principal curvatures.
                                    curvature_ratio = (Tr_H**2) / Det_H

                                    if (Det_H >= 0) and (curvature_ratio < curvature_threshold):
                                        # The ratio of principal curvatures is below the threshold (i.e.,
                                        # it is not an edge point)
                                        curve_keypoints = np.vstack((curve_keypoints, raw_keypoints[-1]))
                                        # Set the loc map to 1 at this point to indicate a keypoint.
                                        loc_per_octave[-1][interval][y, x] = 1
        loc.append(loc_per_octave)

    """
        The next step of the algorithm is to assign orientations to the keypoints.  For this,
        we histogram the gradient orientation over a region about each keypoint.
    """
    from scipy.ndimage.filters import gaussian_filter

    g = gaussian_filter(1.5 * absolute_sigma[0, 3:intervals+3] / subsample[0])
    zero_pad = int(np.ceil(len(g) / 2))

    mag_thresh = np.zeros_like(gauss_pyr)
    mag_pyr = [[None] * (intervals + 3) for _ in range(octaves)]
    grad_pyr = [[None] * (intervals + 3) for _ in range(octaves)]

    for octave in range(octaves):
        for interval in range(2, intervals + 2):
            # Compute x and y derivatives using pixel differences
            diff_x = 0.5 * (gauss_pyr[octave][interval][:, 2:] - gauss_pyr[octave][interval][:, :-2])
            diff_y = 0.5 * (gauss_pyr[octave][interval][2:, :] - gauss_pyr[octave][interval][:-2, :])

            # Compute the magnitude of the gradient
            mag = np.zeros_like(gauss_pyr[octave][interval])
            mag[1:-1, 1:-1] = np.sqrt(diff_x**2 + diff_y**2)

            # Store the magnitude of the gradient in the pyramid with zero padding
            mag_pyr[octave][interval] = np.zeros_like(mag) + np.pad(mag, zero_pad, mode='constant')
            
            # Compute the orientation of the gradient
            grad = np.zeros_like(gauss_pyr[octave][interval])
            grad[1:-1, 1:-1] = np.arctan2(diff_y, diff_x)
            grad[grad == np.pi] = -np.pi

            # Store the orientation of the gradient in the pyramid with zero padding
            grad_pyr[octave][interval] = np.zeros_like(grad) + np.pad(grad, zero_pad, mode='constant')



    return pos, scale, orient, desc