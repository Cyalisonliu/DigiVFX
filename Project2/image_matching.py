import numpy as np
import random

def RANSAC(src_points, dst_points):
    """
    Input:
    src_points - N*2 source pixel location matrices, N is the number of matched keypoints
    dst_points - N*2 destination pixel location matrices, N is the number of matched keypoints
    
    Return:
    best_offset: A list with the current offsets we found between a set of matched keypoints
    """
    threshold = 2.5
    best_offset = []
    max_inliner = 0
    for idx in range(len(dst_points)):
        # calculate offset
        offset = src_points[idx] - dst_points[idx] # next one - last one

        # calculate inliner points
        predicted_pt = src_points - offset  # next one - shift = predicted last one
        differences = dst_points - predicted_pt # last one - predicted last one
        inliner = 0
        for diff in differences:
            if np.sqrt((diff**2).sum()) < threshold: # 2-norm distance
                inliner += 1
        if inliner > max_inliner:
            max_inliner,best_offset = inliner,offset
    return list(best_offset)