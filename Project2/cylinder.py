import cv2 as cv
import numpy as np

# project to a cylinder
def convert(x, y, h, w, F):
    return (F * np.tan((x - w // 2) / F)) + w // 2, ((y - h // 2) / np.cos((x - w // 2) / F)) + h // 2

# read files and set parameters
for i in range(7):
    img = cv.imread(f"./photos/DSC_{6136+i}.JPG")
    h, w, F = img.shape[0], img.shape[1], 8*2316/4.5

    # set the configuration of new image
    new_img = np.zeros(img.shape, dtype=np.uint8)
    new_xy = np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    new_x, new_y = new_xy[:, 0], new_xy[:, 1]
    org_x_flt, org_y_flt = convert(new_x, new_y, h, w, F)
    org_x_int, org_y_int = org_x_flt.astype(int), org_y_flt.astype(int)

    # find valid mask
    valid_mask = (org_x_int >= 0) * (org_x_int <= w - 2) * (org_y_int >= 0) * (org_y_int <= h - 2) 
    new_x, new_y = new_x[valid_mask], new_y[valid_mask]
    org_x_flt, org_y_flt = org_x_flt[valid_mask], org_y_flt[valid_mask]
    org_x_int, org_y_int = org_x_int[valid_mask], org_y_int[valid_mask]

    # derive the weight for points whose coordinates are not integer
    x_err, y_err = org_x_flt - org_x_int, org_y_flt - org_y_int
    top_left, top_right, bottom_left, bottom_right = (1 - x_err) * (1 - y_err), x_err * (1 - y_err), (1 - x_err) * y_err, x_err * y_err
    new_img[new_y, new_x, :] = (top_left[:, None] * img[org_y_int, org_x_int, :]) +\
                            (top_right[:, None] * img[org_y_int, org_x_int + 1, :]) +\
                            (bottom_left[:, None] * img[org_y_int + 1, org_x_int, :]) +\
                            (bottom_right[:, None] * img[org_y_int + 1, org_x_int + 1, :])

    # crop the dark area
    left_edge, right_edge = 0, 0
    for j in range(w):
        if new_img[h // 2][j][0]:
            left_edge = j
            break
    for j in range(w - 1, 0, -1):
        if new_img[h // 2][j][0]:
            right_edge = j
            break
    cropped_img = new_img[:, left_edge:right_edge]

    # output the image
    cv.imwrite(f"./input/img{i}.jpg", cropped_img)