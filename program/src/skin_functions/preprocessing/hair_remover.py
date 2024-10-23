import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_del_paquete)

import cv2 as cv
import numpy as np
import functions.functions as fc


def compute_bottomhat_mask(img, th=50, se_bars=None):
    img_mask = np.zeros(img.shape)

    if se_bars is None:
        se_bars = [fc.se_bar(10, angle) for angle in np.linspace(start=0, stop=180, num=16, endpoint=False)]
    for se_bar in se_bars:
        bottom_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, se_bar)
        img_mask = img_mask + bottom_hat

    img_mask = 255 * (img_mask/np.max(img_mask))
    img_mask = img_mask.astype(np.uint8)
    _, img_mask = cv.threshold(img_mask,th,255,cv.THRESH_BINARY)

    return img_mask


def remove_contours_noise(img_mask, min_area=10, max_circularity=0.3, median_blur=3):
    contours = fc.get_contours(img_mask, filter_area=(0,None))
    contours = fc.filter_contour_circularity(contours, min_circularity=max_circularity)

    img_circular_contours = cv.drawContours(np.zeros_like(img_mask), contours, -1, 255, -1)
    filtered_mask = cv.bitwise_xor(img_mask, img_circular_contours)

    contours = fc.get_contours(filtered_mask, filter_area=(0, min_area))
    img_small_contours = cv.drawContours(np.zeros_like(img_mask), contours, -1, 255, -1)

    filtered_mask = cv.bitwise_xor(filtered_mask, img_small_contours)
    filtered_mask = cv.medianBlur(filtered_mask, median_blur)

    return filtered_mask

def get_hairs_mask(img, se_bars=None):
    img_resized = fc.horizontal_resize(img, w=512)
    img_red = img_resized[:,:,2]
    img_blue_inv = cv.bitwise_not(img_resized[:,:,0])

    img_red = fc.gamma_correction(img_red, 0.5)

    img_blue_inv = fc.gamma_correction(img_blue_inv, 2)
 
    black_hairs_mask = compute_bottomhat_mask(img_red, th=50, se_bars=se_bars)
    white_hairs_mask = compute_bottomhat_mask(img_blue_inv, th=40, se_bars=se_bars)
    black_hairs_mask = remove_contours_noise(black_hairs_mask, min_area=10, max_circularity=0.3)
    white_hairs_mask = remove_contours_noise(white_hairs_mask, min_area=10, max_circularity=0.3)
    img_mask = cv.bitwise_xor(black_hairs_mask, white_hairs_mask)

    img_mask = fc.horizontal_resize(img_mask, img.shape[1], force_h=img.shape[0])
    return img_mask

def remove_hairs(img, se_bars=None):
    img_mask = get_hairs_mask(img, se_bars)
    img_no_hairs = cv.inpaint(img, img_mask, 10, cv.INPAINT_TELEA)
    return img_no_hairs


