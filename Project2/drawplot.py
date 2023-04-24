import numpy as np
import matplotlib.pyplot as plt

def draw_image(img, title):
    print(title)
    plt.imshow(img, cmap='gray')
    plt.show()

def draw_keypoint(img, raw_keypoints, contrast_keypoints, curve_keypoints):
    # Plot keypoints on image
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.plot(raw_keypoints[:, 0], raw_keypoints[:, 1], 'y+')
    ax.set_title('DOG extrema (2x scale)')
    plt.show()
    
    # Plot keypoints after removing low contrast extrema on image
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.plot(contrast_keypoints[:, 0], contrast_keypoints[:, 1], 'y+')
    ax.set_title('Keypoints after removing low contrast extrema (2x scale)')
    plt.show()
    
    # Plot keypoints after removing edge points using principal curvature filtering on image
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.plot(curve_keypoints[:, 0], curve_keypoints[:, 1], 'y+')
    ax.set_title('Keypoints after removing edge points using principal curvature filtering (2x scale)')
    plt.show()

def draw_match_feature(img1, img2 ,match_keypoints):
    # Plot keypoints on LHS image
    fig, ax = plt.subplots()
    ax.imshow(img1, cmap='gray')
    kp1 = match_keypoints[:, 0]
    ax.plot(kp1[:, 0], kp1[:, 1], 'y+')
    ax.set_title('matched keypoints in LHS image')
    plt.show()
    # Plot keypoints on RHS image
    fig, ax = plt.subplots()
    ax.imshow(img2, cmap='gray')
    kp2 = match_keypoints[:, 1]
    ax.plot(kp2[:, 0], kp2[:, 1], 'y+')
    ax.set_title('matched keypoints in RHS image')
    plt.show()

def plot_matches(match_keypoints, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    kp1 = match_keypoints[:, 0]
    kp2 = match_keypoints[:, 1]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    ax.plot(kp1[:, 0], kp1[:, 1], 'xr')
    ax.plot(kp2[:, 0] + offset, kp2[:, 1], 'xr')
    ax.plot([kp1[:, 0], kp2[:, 0] + offset], [kp1[:, 1], kp2[:, 1]],
            'r', linewidth=0.5)
    plt.show()