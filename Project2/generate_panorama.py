import numpy as np
import random


def get_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    # H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    ux = u[:,0].reshape((N,1))
    uy = u[:,1].reshape((N,1))
    vx = v[:,0].reshape((N,1))
    vy = v[:,1].reshape((N,1))

    row_up = np.concatenate( (ux, uy, np.ones((N,1)), np.zeros((N,3)), -ux*vx, -uy*vx, -vx), axis=1 );
    row_down = np.concatenate( (np.zeros((N,3)), ux, uy, np.ones((N,1)), -ux*vy, -uy*vy, -vy), axis=1 );
    A = np.concatenate( (row_up, row_down), axis=0 )

    U, S, VT = np.linalg.svd(A)
    H = VT[-1,:]/VT[-1,-1]
    H = H.reshape(3, 3)
    return H

def RANSAC(src_points, dst_points, last_best_H):
    # print(src_points)
    # print(dst_points.shape)
    times = 5000
    threshold = 4
    inline_max = 0
    H_best = np.eye(3)
    for _ in range(times):
        src = np.zeros((4,2))
        dst = np.zeros((4,2)) 
        # randomly generate 4 points to find H
        for j in range(4):
            idx = random.randint(0, len(src_points)-1)
            src[j] = src_points[idx]
            dst[j] = dst_points[idx]
        H = get_homography(src, dst)
        
        constant = np.ones((1,len(src_points)))
        # 2D -> 3D
        M = np.concatenate( (np.transpose(dst), np.ones((1,dst.shape[0]))), axis=0)
        # get Mbar after transfer M which represents the position of dst points  
        Mbar = np.dot(H,M)
        # 3D -> 2D
        Mbar = np.divide(Mbar, Mbar[-1,:])

        W = np.concatenate( (np.transpose(src), np.ones((1,src.shape[0]))), axis=0)   
        # print('Mbar{}'.format(Mbar))   
        # print('W{}'.format(W))
        
        diff = Mbar[:-1][:] - W[:-1][:]
        # print('diff{}'.format(diff))
        err = np.linalg.norm(diff, axis=0)
        inline_cnt = sum(err<threshold)
        inline_src = src[err<threshold]
        inline_dst = dst[err<threshold]
        
        if inline_cnt > inline_max:
            inline_max = inline_cnt
            H_best = H

    last_best_H = last_best_H.dot(H_best)
    # output = warping(im2, dst, last_best_H, 0, im2.shape[0], w, w+im2.shape[1], direction='b') 

def panorama(image_list):
    last_best_H = np.eye(3)
    RANSAC()
    return 