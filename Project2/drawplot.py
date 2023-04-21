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