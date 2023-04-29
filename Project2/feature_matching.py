import numpy as np
from sklearn.neighbors import KDTree

def kd_tree_matching(img1, kp1, des1, kp2, des2):
    """
    Input:
    img1 - array of the image on the LHS. Also seen as destination image
    kp1 - N*2 keypoints array, N is the number of keypoints of the image on the LHS we found in SIFT
    des1 - N*128 descriptors array, N is the number of keypoints of the image on the LHS we found in SIFT
    kp2 - N*2 keypoints array, N is the number of keypoints of the image on the RHS we found in SIFT 
    des2 - N*128 descriptors array, N is the number of keypoints of the image on the RHS we found in SIFT

    Return:
    matched_pairs: An array with the matched keypoints we found in RANSAC
    """
    dist_threshold = 0.25
    h, w = img1.shape
    tree = KDTree(des2, leaf_size=2)
    dist, indice2 = tree.query(des1, k=3)
    dist = dist[:, 0].reshape(-1,).tolist()
    indice2 = indice2[:, 0].reshape(-1,).tolist()
    indice1 = [i for i in range(len(dist))]
    indice1.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indice1] #sorted dist
    indice2 = [indice2[i] for i in indice1] #sorted idx for the image on the rhs

    purified_kp1, purified_kp2 = [], []
    for idx, dis in zip(indice2, dist):
        if dis < dist_threshold:
            purified_kp2.append(kp2[idx])
        else:
            break

    for idx, dis in zip(indice1, dist):
        if dis < dist_threshold:
            purified_kp1.append(kp1[idx])
        else:
            break

    # Assuming that the pictures are taken from left to right
    matched_pairs = []
    x_thres = w / 10
    y_thres = h / 10

    for i in range(len(purified_kp2)):
        distance_x = purified_kp1[i][0] - purified_kp2[i][0]
        distance_y = abs(purified_kp1[i][1] - purified_kp2[i][1])
        if distance_y < y_thres and distance_x < w and distance_x > x_thres:
            feat_pt1 = np.array([int(purified_kp1[i][0]), int(purified_kp1[i][1])])
            feat_pt2 = np.array([int(purified_kp2[i][0]), int(purified_kp2[i][1])]) # x, y
            matched_pairs.append([feat_pt1, feat_pt2])

    return np.asarray(matched_pairs)