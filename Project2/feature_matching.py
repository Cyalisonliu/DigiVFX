import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree

def kd_tree_matching(img1, img2, kp1, des1, kp2, des2):
    kp_max_cnt = kp2.shape[0]
    print(kp1.shape, kp2.shape)
    cutoff = 0.25
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
        if dis < cutoff:
            purified_kp2.append(kp2[idx])
        else:
            break

    for idx, dis in zip(indice1, dist):
        if dis < cutoff:
            purified_kp1.append(kp1[idx])
        else:
            break

    # Assuming that the pictures are taken from left to right
    matched_pairs = []
    matched_x_thres = w / 10
    matched_y_thres = h / 20

    for i in range(len(purified_kp2)):
        distance_x = purified_kp1[i][0] - purified_kp2[i][0]
        distance_y = abs(purified_kp1[i][1] - purified_kp2[i][1])
        if distance_y < matched_y_thres and distance_x < w and distance_x > matched_x_thres:
            feat_pt1 = np.array([int(purified_kp1[i][0]), int(purified_kp1[i][1])])
            feat_pt2 = np.array([int(purified_kp2[i][0]), int(purified_kp2[i][1])]) # x, y
            print(feat_pt1, feat_pt2)
            matched_pairs.append([feat_pt1, feat_pt2])

    return np.asarray(matched_pairs)