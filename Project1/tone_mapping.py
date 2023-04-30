import numpy as np
import cv2

def Reinhard_tonemap(img, delta, alpha, L_white, standard_color_img):
    print("processing tonemapping...")
    h, w, rgb = img.shape
    B, G, R = 0, 1, 2
    Lw = np.transpose(img, (2, 0, 1))
    L = np.zeros_like(Lw)
    Ld = np.zeros_like(Lw)
    
    for i in range(rgb):
        L_bar = np.exp(np.sum(np.log(delta+Lw[i])) / (h*w))
        L[i] = Lw[i] * alpha / L_bar
        Ld[i] = (L[i] * (1+(L[i]/(L_white**2)))) / (1+L[i])

    L_output = np.zeros((rgb, h, w))
    L_output[B] = Ld[2]
    L_output[G] = Ld[1]
    L_output[R] = Ld[0]
    L_output = np.transpose(L_output, (1, 2, 0))
    L_output *= 255

    adjust_output = np.zeros_like(L_output)
    for i in range(rgb):
        adjust_output[:,:,i] = L_output[:,:,i]*(np.average(standard_color_img[:,:,i])/np.average(L_output[:,:,i]))
    return adjust_output
    