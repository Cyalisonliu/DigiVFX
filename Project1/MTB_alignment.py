import cv2
import numpy as np

def convert_gray(imgs):
    b, g, r = imgs[:,:,0], imgs[:,:,1], imgs[:,:,2]
    gray_img = (54*r+183*g+19*b) / 256
    return gray_img

def get_bitmap(img, idx, noise=4, _print=False):
    img_median = np.median(img)
    threshold_upper = img_median + noise
    threshold_lower = img_median - noise
    e_bitmap = np.where(img > threshold_upper, 1, 0) + np.where(img < threshold_lower, 1, 0)
    if _print:
        h, w = e_bitmap.shape
        image_test = np.zeros((h, w, 3))
        image_test[:,:,0] = e_bitmap*255
        image_test[:,:,1] = e_bitmap*255
        image_test[:,:,2] = e_bitmap*255
        cv2.imwrite(f'./e_bitmap{idx}.jpg', image_test.astype(np.float32))
    return e_bitmap
    
def generate_pyramids(imgs, num_level):
    pyramid_imgs = []
    for idx, img in enumerate(imgs):
        bit_img = get_bitmap(img, idx, 2, False)
        pyramid_per_img = [bit_img]
        for i in range(num_level-1):
            base_img = pyramid_per_img[-1].astype(np.float32)
            pyramid_per_img.append(cv2.resize(base_img, (int(base_img.shape[1]/2), int(base_img.shape[0]/2)), interpolation=cv2.INTER_NEAREST))
        pyramid_imgs.append(pyramid_per_img)
    return pyramid_imgs

def get_best_shift(stdandard_img, cur_img, shift):
    min_diff = float('Inf')
    best_shift = np.array([0, 0])
    for s in shift:
        align_img = shift_img(np.uint8(cur_img), s, False)
        dif_XOR = np.sum((np.logical_xor(stdandard_img, align_img)))
        if dif_XOR < min_diff:
            best_shift = s
            min_diff = dif_XOR
    # print(best_shift)
    return best_shift

def shift_img(img, shift, _print=False):
    M = np.float32([[1, 0, shift[0]],[0, 1, shift[1]]])
    shifted_image = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    if _print :
        cv2.imwrite(f'./test_shift{[shift[0],shift[1]]}.jpg', shifted_image)
    return shifted_image

def calculate_offset(stdandard_imgs, pyramid_imgs, num_level):
    direction = [ (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    best_shift = []
    for pyramid_img in pyramid_imgs:
        offset = np.array([0, 0])
        for i in range(num_level-1, -1, -1): # for each size
            stdandard_img = stdandard_imgs[i]
            cur_img = pyramid_img[i]
            shift = [[2*offset[0]+direction[d][0], 2*offset[1]+direction[d][1]] for d in range(9)]
            offset = get_best_shift(stdandard_img, cur_img, shift)
        best_shift.append(offset)
    print("best shift found!\n",best_shift)
    return np.array(best_shift)

def MTB(imgs, standard_idx, num_level): 
    gray_imgs = []
    for img in imgs:
        gray_imgs.append(convert_gray(img))
    gray_imgs = np.array(gray_imgs)
    pyramid_imgs = generate_pyramids(gray_imgs, num_level)
    stdandard_imgs = pyramid_imgs[standard_idx]
    offsets = calculate_offset(stdandard_imgs, pyramid_imgs, num_level)        
        
    return offsets
