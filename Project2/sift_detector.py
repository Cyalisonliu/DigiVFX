import cv2
from scipy.ndimage import gaussian_filter

from utilis import generate_pyramid, get_keypoints, assign_orientation, generate_descriptor
from drawplot import draw_image, draw_keypoint, draw_final_kpts


def SIFT_get_features(img, draw, s=3, num_octave=4, sigma=1.6, curvature_threshold=10.0, contrast_threshold=3.5):
    # Blur the image with a standard deviation of 0.5
    # Upsample the image by a factor of 2 using linear interpolation
    antialias_sigma = 0.5
    if antialias_sigma == 0:
        signal = img
    else:
        signal = gaussian_filter(img, sigma=antialias_sigma, mode='constant')

    # Upsample the image by a factor of 2 using linear interpolation
    h, w = signal.shape
    signal = cv2.resize(signal, (int(w*2), int(h*2)), interpolation=cv2.INTER_NEAREST)
    subsample = [0.5]
    curvature_threshold = (curvature_threshold+1)**2 / curvature_threshold

    """
        1. Generate gaussain pyrimid and DoG pyrimid
    """
    gaussian_pyr, DOG_pyr, subsample = generate_pyramid(signal, num_octave, s, sigma, subsample)
    if draw:
        for i, gaussian_per in enumerate(gaussian_pyr[1]):
            draw_image(gaussian_per, f'Gaussain image {i+1} 1x')
        for i, dog_per in enumerate(DOG_pyr[1]):
            draw_image(dog_per, f'DOG image {i+1} 1x')

    """
        2. Detect keypoints over DoG pyrimid
    """
    kp_pyr, raw_keypoints, contrast_keypoints, curve_keypoints = get_keypoints(DOG_pyr, num_octave, s, subsample, contrast_threshold, curvature_threshold)
    if draw:
        draw_keypoint(img, raw_keypoints, contrast_keypoints, curve_keypoints)

    """
        3. Assign orientations to the keypoints and get descriptor
    """
    kp_pyr, orient, scale = assign_orientation(kp_pyr, gaussian_pyr, s, num_octave, subsample)
    kp_pos, descriptor = generate_descriptor(kp_pyr, gaussian_pyr, orient, scale, subsample)
    if draw:
        draw_final_kpts(img, kp_pyr)

    return kp_pos, descriptor