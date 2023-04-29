import numpy as np
import cv2

def stitching(img_path, offsets):
    img_list = [cv2.imread(path) for path in img_path]
    dst_h, dst_w, _ = img_list[0].shape

    x_offest = 0
    y_offest_max = 0
    y_offest_min = float("inf")
    for offset in offsets:
        x_offest += offset[0]
        if offset[1] < y_offest_min and offset[1] >= 0: 
            y_offest_min = offset[1]
        if offset[1] < 0: 
            y_offest_max += abs(offset[1])

    result_image = np.zeros((dst_h+abs(y_offest_min)+abs(y_offest_max), dst_w+abs(x_offest), 3))
    result_image[abs(y_offest_min): abs(y_offest_min)+dst_h, :dst_w] = img_list[0]
    prev_offset = np.array([0,0])
    width = dst_w
    # Each time, update previous image and paste the next image
    for i, (img1, img2) in enumerate(zip(img_list[:], img_list[1:])):
        final_offset = np.array([0,0])
        for offset in offsets[:i+2]: 
            final_offset += offset
        prev_pixels, prev_h, prev_w = img1, img1.shape[0], img1.shape[1]
        cur_pixels, cur_h, cur_w = img2, img2.shape[0], img2.shape[1]
        overlap = width - abs(final_offset[0])
        start_left = prev_w-overlap
        prev_overlap = np.zeros((prev_h, overlap, 3))
        for pix in range(0, overlap):
            cur_pixels[:,pix,:] = cur_pixels[:,pix,:] * (pix / overlap)
            prev_overlap[:,pix,:] = prev_pixels[:,start_left+pix,:] * ((overlap-1-pix) / overlap)
        result_image[abs(y_offest_min)-prev_offset[1]:abs(y_offest_min)-prev_offset[1]+prev_h, abs(final_offset[0]):abs(final_offset[0])+overlap,:] = prev_overlap
        result_image[abs(y_offest_min)-final_offset[1]:abs(y_offest_min)-final_offset[1]+cur_h, abs(final_offset[0]):abs(final_offset[0])+cur_w, :] += cur_pixels
        width = cur_w + abs(final_offset[0])
        prev_offset = final_offset

    # without blending
    # for i, (img1, img2) in enumerate(zip(img_list[:], img_list[1:])):
    #     final_offset = np.array([0,0])
    #     for offet in offsets[:i+2]: 
    #         final_offset += offet
    #     prev_pixels, prev_h, prev_w = img1, img1.shape[0], img1.shape[1]
    #     cur_pixels, cur_h, cur_w = img2, img2.shape[0], img2.shape[1]
    #     overlap = width - abs(final_offset[0])
    #     result_image[abs(y_offest_min)-final_offset[1]:abs(y_offest_min)-final_offset[1]+cur_h, abs(final_offset[0]):abs(final_offset[0])+cur_w, :] = cur_pixels
    #     width = cur_w + abs(final_offset[0])
    #     prev_offset = final_offset
    
    # crop image
    checkrange = 10
    r_h, r_w, _ = result_image.shape 
    left = 0
    right = r_w
    top = 0
    bottom = r_h
    for pixy in range(r_h//2):
        if np.max(result_image[pixy, -1]):
            if pixy > top: 
                top = pixy
                break
    for pixy in range(r_h//2):
        if np.max(result_image[r_h-1-pixy, 0]):
            if r_h-1-pixy < bottom: 
                bottom = r_h-1-pixy
                break
    
    crop_img = result_image[top+checkrange:bottom-checkrange, left:right]

    return result_image, crop_img
