import numpy as np

def RANSAC(src_points, dst_points):
    threshold = 2.5
    best_offset = []
    max_inliner = 0
    for idx in range(len(dst_points)):
        offset = src_points[idx] - dst_points[idx]
        pred_pos = src_points - offset
        diff = dst_points - pred_pos
        inliner = sum(np.linalg.norm(d) < threshold for d in diff)
        if inliner > max_inliner:
            max_inliner, best_offset = inliner, offset
    return list(best_offset)