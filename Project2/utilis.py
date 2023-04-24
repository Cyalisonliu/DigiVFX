import numpy as np
import cv2
import math
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

from drawplot import draw_image

"""
Define gaussain filter
"""
def get_gaussain_filter(sigma):
    ksize = 2 * math.ceil(2 * sigma) + 1
    half_size = int(ksize / 2)
    kernel_2d = np.zeros((ksize, ksize))
    for i in range(ksize):
        for j in range(ksize):
            offset = (i - half_size, j - half_size)
            kernel_2d[i, j] = np.exp(-(offset[0] ** 2 + offset[1] ** 2) / (2 * sigma ** 2))
    kernel_2d /= np.sum(kernel_2d)
    return kernel_2d

"""
Code for generating Gaussain pyrimid & DoG pyrimid
"""
def generate_octave(first_img, s, sigma):
    k = 2**(1/s)
    DOG_per_octave = []
    gaussain_per_octave = [gaussian_filter(first_img, sigma=sigma, mode='constant')]

    for i in range(1, s+1):
        gaussain_per_octave.append(gaussian_filter(gaussain_per_octave[-1], sigma=(k**i)*sigma, mode='constant'))
        DOG_per_octave.append(gaussain_per_octave[i]-gaussain_per_octave[i-1])
    """
    (4, 512, 512)
    (4, 256, 256)
    (4, 128, 128)
    (4, 64, 64)
    """
    return np.asarray(gaussain_per_octave, dtype='float32'), np.asarray(DOG_per_octave, dtype='float32')

def generate_pyramid(base_img, num_octave, s, sigma, subsample):
    gaussain_pyr = []
    DOG_pyr = []
    
    for i in range(num_octave):
        gaussain_per_octave,  dog_per_octave = generate_octave(base_img, s, sigma)
        gaussain_pyr.append(gaussain_per_octave)
        DOG_pyr.append(dog_per_octave)
        h, w = gaussain_per_octave[-1].shape
        base_img = cv2.resize(gaussain_per_octave[-1], (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        # base_img = gaussain_per_octave[-1].resize((w//2, h//2), Image.NEAREST)
        if i > 0:
            subsample.append(subsample[-1]*2)

    return gaussain_pyr, DOG_pyr, subsample

"""
Code for detecting keypoints
"""
def computeHessianAtCenterPixel(pixel_array):
    """
    Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

def get_keypoints(DOG_pyr, num_octave, s, subsample, contrast_threshold, curvature_threshold):
    # 2nd derivative kernels
    xx = np.array([1, -2, 1]) / 2
    yy = xx.reshape(-1, 1)
    xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4

    # Coordinates of keypoints after each stage of processing for display in interactive mode.
    raw_keypoints = []
    contrast_keypoints = []
    curve_keypoints = []

    # Detect local maxima in the DOG pyramid
    loc_map = [np.zeros_like(DOG_pyr[i]) for i in range(len(DOG_pyr))]  # boolean maps of keypoints
    for octave in range(num_octave):    
        for layer in range(1, s-1):
            # contrast_mask = np.abs(DOG_pyr[octave][layer, :, :]) >= contrast_threshold
            # loc_map[octave][layer] = np.zeros_like(DOG_pyr[octave][layer, :, :])
 
            for y in range(1, DOG_pyr[octave][layer].shape[0]-1):
                for x in range(1, DOG_pyr[octave][layer].shape[1]-1):
                    # Only check for extrema where the object mask is 1
                    # Check for a max or a min across space and scale
                    window = DOG_pyr[octave][layer-1:layer+2, y-1:y+2, x-1:x+2]
                    center = window[1, 1, 1]
                    if ((center >= np.amax(window[0])) and (center >= np.amax(window[1])) and (center >= np.amax(window[2])))\
                         or ((center <= np.amin(window[0])) and (center <= np.amin(window[1])) and (center <= np.amin(window[2]))):
                        raw_keypoints.append([x*subsample[octave], y*subsample[octave]])
                        if np.abs(center) >= contrast_threshold:
                            # print(center)
                            contrast_keypoints.append(raw_keypoints[-1])
                            # Compute the entries of the Hessian matrix at the extrema location.
                            hessian = computeHessianAtCenterPixel(np.array(window))
                            xy_hessian = hessian[:2, :2]
                            Tr_H = np.trace(xy_hessian)
                            Det_H = np.linalg.det(xy_hessian)
                            # Dxx = np.sum(np.multiply(window[1, 0:3], xx))
                            # Dyy = np.sum(np.multiply(window[0:3, 1], yy))
                            # Dxy = np.sum(np.sum(np.multiply(window[0:3, 0:3], xy)))

                            # Compute the trace and the determinant of the Hessian.
                            # Tr_H = Dxx + Dyy
                            # Det_H = Dxx*Dyy - Dxy**2

                            # Compute the ratio of the principal curvatures.
                            if Det_H > 0:
                                curvature_ratio = (Tr_H**2) / Det_H
                                if curvature_ratio < curvature_threshold:
                                    # it is not an edge point
                                    curve_keypoints.append(raw_keypoints[-1])
                                    # Set the loc map to 1 at this point to indicate a keypoint.
                                    loc_map[octave][layer][y,x] = 1
    print(len(raw_keypoints), len(contrast_keypoints), len(curve_keypoints))
    return loc_map, np.array(raw_keypoints), np.array(contrast_keypoints), np.array(curve_keypoints)

"""
Code for assigning orientation to keypoints
"""
def assign_orientation(kp_pyr, gaussain_pyr, s, num_octave, subsample):
    mag_pyr = [np.zeros_like(gaussain_pyr[i]) for i in range(len(gaussain_pyr))]
    grad_pyr = [np.zeros_like(gaussain_pyr[i]) for i in range(len(gaussain_pyr))] 
    # zero_pad = 1
    num_bins = 36
    hist_step = 2*np.pi / num_bins
    hist_orient = np.arange(-np.pi, np.pi, hist_step)
    pos = []
    orient = []
    scale = []
    for octave in range(num_octave):    
        for layer in range(1, s-1):
            kp_map = np.asarray(kp_pyr[octave][layer])
            gauss_pyr = np.asarray(gaussain_pyr[octave][layer])
            # Compute x and y derivatives using pixel differences
            diff_x = 0.5 * (gauss_pyr[1:-1, 2:] - gauss_pyr[1:-1, :-2])
            diff_y = 0.5 * (gauss_pyr[2:, 1:-1] - gauss_pyr[:-2, 1:-1])
            # Compute the magnitude of the gradient
            mag = np.zeros_like(gauss_pyr)
            mag[1:-1,1:-1] = np.sqrt(diff_x**2 + diff_y**2)
            mag_pyr[octave][layer,:,:] = mag
            # Compute the orientation of the gradient
            grad = np.zeros_like(gauss_pyr)
            grad[1:-1,1:-1] = np.arctan2(diff_y, diff_x)
            grad[np.where(grad == np.pi)] = -np.pi
            grad_pyr[octave][layer,:,:] = grad
            
            # Assign orientations to the keypoints
            # Get gaussian filter and gaussain filter size
            # sigma = 1.5*scale
            sigma = 1.5/subsample[octave]
            ksize = 2 * math.ceil(2 * sigma) + 1
            g = get_gaussain_filter(sigma)
            hf_sz = int(ksize / 2)

            """
                Iterate over all the keypoints at this octave and orientation.
                Histogram the gradient orientations for this keypoint weighted by the
                gradient magnitude and the gaussian weighting mask.
            """
            ixy = np.where(kp_map == 1)
            iy = list(ixy[0])
            ix = list(ixy[1])
            for k in range(len(iy)):
                y = iy[k]
                x = ix[k]
                h, w = mag_pyr[octave][layer].shape
                if (y - hf_sz < 0 or y + hf_sz + 1 >= h or x - hf_sz < 0 or x + hf_sz + 1 >= w):
                    continue
                # print(g.shape, y - hf_sz, y + hf_sz + 1, x - hf_sz, x + hf_sz + 1)
                # print(mag_pyr[octave][layer].shape)
                weight = g * mag_pyr[octave][layer][y - hf_sz : y + hf_sz + 1, x - hf_sz : x + hf_sz + 1]
                grad_window = grad_pyr[octave][layer][y - hf_sz : y + hf_sz + 1, x - hf_sz : x + hf_sz + 1]
                orient_hist = np.zeros(len(hist_orient))
                for hist_bin in range(len(hist_orient)):
                    # Compute the difference of the orientations mod pi
                    diff = np.mod(grad_window - hist_orient[hist_bin] + np.pi, 2 * np.pi) - np.pi
                    # Accumulate the histogram bins
                    """
                        np.abs(diff) / hist_step -> 梯度方向差值和bins長度的比值
                        代表了梯度方向差值落在此hist_bin的程度
                        越接近 1 -> 差值越小
                        越接近 0 -> 差值越大 -> 可能落在此直方圖區間
                    """
                    orient_hist[hist_bin] += np.sum(weight * np.maximum(1 - np.abs(diff) / hist_step, 0))
                # Find peaks in the orientation histogram using nonmax suppression.
                peaks = orient_hist.copy()
                # Concate the start of the histogram and the end of the histogram
                rot_right = np.concatenate((peaks[-1:], peaks[:-1]))
                rot_left = np.concatenate((peaks[1:], peaks[:1]))
                # If the value is smaller than its right bin or left bin, it won't be a peak value
                peaks[np.where(peaks < rot_right)] = 0
                peaks[np.where(peaks < rot_left)] = 0
                # Extract the value and index of the largest peak.
                peak_max = np.max(peaks)
                peak_idx = np.argmax(peaks)

                # Iterate over all peaks within 80% of the largest peak and add keypoints with
                # the orientation corresponding to those peaks to the keypoint list.
                peak_value = peak_max
                while peak_value > 0.8 * peak_max:
                    # Interpolate the peak by fitting a parabola to the three histogram values closest to each peak.
                    A = np.zeros((3, 3))
                    b = np.zeros((3, 1))
                    for j in range(-1, 2):
                        A[j+1, :] = [((hist_orient[peak_idx] + hist_step*j) ** 2),
                                    (hist_orient[peak_idx] + hist_step*j),
                                    1]
                        hist_bin = np.mod(peak_idx + j + num_bins - 1, num_bins)
                        b[j+1] = orient_hist[hist_bin]
                    # solve theta in A theta = b
                    theta = np.linalg.pinv(A) @ b
                    max_orient = -theta[1] / (2*theta[0])
                    while max_orient < -np.pi:
                        max_orient = max_orient + 2 * np.pi
                    while max_orient >= np.pi:
                        max_orient = max_orient - 2 * np.pi

                    # Store the keypoint position, orientation, and scale information
                    pos.append([x* subsample[octave], y* subsample[octave]])
                    orient.append(max_orient)
                    scale.append([octave, layer, sigma])
                    # next peak
                    peaks[peak_idx] = 0
                    peak_value, peak_idx = np.max(peaks), np.argmax(peaks)

    for i in range(len(pos)):
        print(i, np.asarray(pos[i]), orient[i], scale[i][2])

    # print(len(kp_pyr[0][0]), len(grad_pyr[0][0])) shape is the same
    return np.asarray(pos), np.asarray(orient), np.asarray(scale)
"""
Code to extract feature descriptors for the keypoints.
The descriptors are a grid of gradient orientation histograms, where the sampling
grid for the histograms is rotated to the main orientation of each keypoint.  The
grid is a 4x4 array of 4x4 sample cells of 8 bin orientation histograms.  This 
procduces 128 dimensional feature vectors.
"""
def generate_descriptor(kp_pos, gaussain_pyr, orient, scale, subsample):
    # The orientation histograms have 8 bins, each bin has pi/4 size
    orient_bin_spacing = np.pi / 4
    orient_angles = np.arange(-np.pi, np.pi, orient_bin_spacing)
    print(orient_angles)

    # The feature grid is has 4x4 cells - feat_grid describes the cell center positions
    grid_spacing = 4
    x_coords, y_coords = np.meshgrid(np.arange(-6, 7, grid_spacing), np.arange(-6, 7, grid_spacing))
    feat_grid = np.array((x_coords.flatten(), y_coords.flatten()))
    x_coords, y_coords = np.meshgrid(np.arange(-(2*grid_spacing-0.5), 2*grid_spacing+0.5), np.arange(-(2*grid_spacing-0.5), 2*grid_spacing+0.5))
    feat_samples = np.array((x_coords.flatten(), y_coords.flatten()))
    feat_window = 2 * grid_spacing

    # Initialize the descriptor list to the empty matrix.
    desc = []
    # look over all keypoints
    for k in range(kp_pos.shape[0]):
        x = kp_pos[k, 0] / subsample[int(scale[k, 0])]
        y = kp_pos[k, 1] / subsample[int(scale[k, 0])]
        # Rotate the grid coordinate
        # np.dot(M, feat_grid) -> rotate, + np.tile -> move to right position
        M = np.array([[math.cos(orient[k]), -math.sin(orient[k])],
            [math.sin(orient[k]), math.cos(orient[k])]])
        feat_rotate_grid = np.dot(M, feat_grid) + np.tile(np.array([[x],[y]]), (1, feat_grid.shape[1]))
        feat_rotate_samples = np.dot(M, feat_samples) + np.tile(np.array([[x],[y]]), (1, feat_samples.shape[1]))
        # print(feat_rotate_grid.shape, feat_rotate_samples.shape)
        # Initialize the feature descriptor
        feat_desc = np.zeros((128,))

        # look over all the samples in the sampling grid    
        for s_idx in range(feat_rotate_samples.shape[1]):
            x_sample = feat_rotate_samples[0, s_idx]
            y_sample = feat_rotate_samples[1, s_idx]
            # Interpolate the gradient at the sample position (surrounding sample pos)
            X, Y = np.meshgrid(np.arange(x_sample-1, x_sample+2), np.arange(y_sample-1, y_sample+2))
            G = map_coordinates(gaussain_pyr[int(scale[k,0])][int(scale[k,1])], [Y, X], order=1, prefilter=False)
            G[np.isnan(G)] = 0
            diff_x = 0.5 * (G[1,2] - G[1,0])
            diff_y = 0.5 * (G[2,1] - G[0,1])
            mag_sample = np.sqrt(diff_x**2 + diff_y**2)
            grad_sample = np.arctan2(diff_y, diff_x)
            if grad_sample == np.pi:
                grad_sample = -np.pi

            # Compute the weighting for the x and y dimensions.
            x_weight = np.maximum(1 - (np.abs(feat_rotate_grid[0, :] - x_sample) / grid_spacing), 0)
            y_weight = np.maximum(1 - (np.abs(feat_rotate_grid[1, :] - y_sample) / grid_spacing), 0)
            pos_weight = np.reshape(np.tile(x_weight * y_weight, (8, 1)), (1, 128))
            # Compute the weighting for the orientation, rotating the gradient to the
            # main orientation to of the keypoint first, and then computing the difference
            # in angle to the histogram bin mod pi.
            diff = np.mod(grad_sample - orient[k] - orient_angles + np.pi, 2 * np.pi) - np.pi
            orient_weight = np.maximum(1 - np.abs(diff) / orient_bin_spacing, 0)
            orient_weight = np.tile(orient_weight, (1, 16))

            # Compute the gaussian weighting.
            offset = (x_sample - x, y_sample - y)
            g = np.exp(-(offset[0]**2 + offset[1]**2) / (2*feat_window**2)) / (2*np.pi*feat_window**2)

            # Accumulate the histogram bins.
            feat_desc += pos_weight[0] * orient_weight[0] * g * mag_sample
        # Normalize the feature descriptor to a unit vector to make the descriptor invariant to affine changes in illumination.
        feat_desc /= np.linalg.norm(feat_desc)
        # Threshold the large components in the descriptor to 0.2 and then renormalize 
        # to reduce the influence of large gradient magnitudes on the descriptor.
        feat_desc[np.where(feat_desc > 0.2)] = 0.2
        feat_desc /= np.linalg.norm(feat_desc)
        desc.append(feat_desc)

    # Adjust for the sample offset
    for k in range(kp_pos.shape[0]):
        kp_pos[k,:] -= subsample[int(scale[k,0])]-1

    # Return only the absolute scale
    # if kp_pos.shape[0] > 0:
    #     scale = scale[:,2]

    return kp_pos, np.array(desc)
