import numpy as np
import cv2
from PIL import Image

def stitching(img_path, offsets):
    img_list = [Image.open(path) for path in img_path]
    dst_h, dst_w, _ = np.array(img_list[0]).shape

    x_offest = 0
    y_offest_max = -1*float("inf")
    y_offest_min = float("inf")
    for offest in offsets:
        x_offest += offest[0]
        if offest[1] < y_offest_min and offest[1] >= 0: 
            y_offest_min = offest[1]
        if offest[1] > y_offest_max and offest[1] < 0: 
            y_offest_max = offest[1]

    result_image = Image.new(mode="RGB", size=(dst_w+abs(x_offest), dst_h+abs(y_offest_min)+abs(y_offest_max)))
    # store current width of the result picture
    width = 0

    for i,img in enumerate(img_list):
        final_offset = np.array([0,0])
        for offet in offsets[:i+1]: 
            final_offset += offet
        if final_offset[0] != 0:
            # all the image causing overlapping
            # calculate the region two pictures overlap
            overlap = width - abs(final_offset[0])
            pixelrange_left = range(img_list[i-1].size(0) - overlap ,img_list[i-1].size(0))
            pixelrange_right = range(0, overlap)
            ### TODO ###
            """
            Linear blending on img_list[i-1] and img
            """
            # update current width
            width = img.size(0) + abs(final_offset[0])
        else:
            # first image
            result_image.paste(img, (abs(final_offset[0]), abs(y_offest_min)-final_offset[1]))
            width = img.size(0) + abs(final_offset[0])
    result_image.save('final.jpg')

    return result_image
