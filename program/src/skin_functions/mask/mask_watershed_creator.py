import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_del_paquete)

import cv2 as cv
import numpy as np
import functions.functions as fc


def get_lession_and_skin_mask(img_mean_b_r, kernel_l_size=9, kernel_s_size=51):
    # obtain lession and skin masks
    _, mask_lesion = cv.threshold(img_mean_b_r, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    mask_skin = cv.bitwise_not(mask_lesion)

    kernel_l = np.ones((kernel_l_size, kernel_l_size), np.uint8)
    mask_lesion = cv.erode(mask_lesion, kernel_l, iterations=1)

    kernel_s = np.ones((kernel_s_size, kernel_s_size), np.uint8)
    mask_skin = cv.erode(mask_skin, kernel_s, iterations=1)
    return mask_lesion, mask_skin


def get_markers(img_mean_b_r, kernel_l_size=9, kernel_s_size=51):
    mask_lesion, mask_skin = get_lession_and_skin_mask(img_mean_b_r, kernel_l_size, kernel_s_size)

    # create markers
    markers = np.zeros(img_mean_b_r.shape[:2], dtype=np.int32)
    markers[mask_lesion > 0] = 1  
    markers[mask_skin > 0] = 2 

    return markers


def get_mask(img, kernel_l_size=9, kernel_s_size=51, show=False, img_nb=0, img_label=None, resize_w=512):
    # reduce the resolution to be faster
    img_resized = fc.horizontal_resize(img, resize_w)

    # obtaine intensity image
    img_mean_b_r = fc.compute_image_mean_channels(img_resized, channels=[0,2], importance=[1,1])
    markers = get_markers(img_mean_b_r, kernel_l_size, kernel_s_size)

    # apply watershed
    img_water_shed = np.zeros_like(img_resized)
    img_water_shed[:,:,0] = img_mean_b_r
    img_water_shed[:,:,1] = img_mean_b_r
    img_water_shed[:,:,2] = img_mean_b_r
    cv.watershed(img_water_shed, markers)

    # create mask
    mask = np.zeros(img_resized.shape[:2], dtype=np.uint8)
    mask[markers==1] = 255

    # resize mask back to original dimension
    mask = fc.horizontal_resize(mask, img.shape[1], force_h=img.shape[0], interpolation=cv.INTER_NEAREST)

    return mask

