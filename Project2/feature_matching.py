import numpy as np
from sklearn.neighbors import KDTree

def kd_tree_matching(img1, kp1, des1, kp2, des2):
    h, w = img1.shape
    x_thres = w / 10
    y_thres = h / 10
    dist_threshold = 0.25
    tree = KDTree(des2, leaf_size=2)
    dist, indice2 = tree.query(des1, k=3)
    dist = dist[:, 0].reshape(-1,).tolist() # only use nearest one
    indice2 = indice2[:, 0].reshape(-1,).tolist()
    indice1 = [i for i in range(len(dist))]
    indice1.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indice1]
    indice2 = [indice2[i] for i in indice1]
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

    matched_pairs = []
    for i in range(len(purified_kp2)):
        dist_x = purified_kp1[i][0] - purified_kp2[i][0] # must > 0 because we go from left
        dist_y = abs(purified_kp1[i][1] - purified_kp2[i][1])
        if dist_y < y_thres and dist_x < w and dist_x > x_thres:
            matched_pairs.append([np.array([int(purified_kp1[i][0]), int(purified_kp1[i][1])]), np.array([int(purified_kp2[i][0]), int(purified_kp2[i][1])])])

    return np.array(matched_pairs)