import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_del_paquete)

import cv2 as cv
import numpy as np
import functions.functions as fc



def remove_border_contours_and_fill_holes(img_mask, medianb_size=7, gaussian_factor=6, safp=99/100, resize_w=512):
    mask = fc.horizontal_resize(img_mask, resize_w, interpolation=cv.INTER_NEAREST)
    
    if medianb_size > 0:
        mask = cv.medianBlur(mask, medianb_size)

    contours = fc.get_contours(mask, order_by_area=False, filter_area=(10, None))
    if len(contours) == 0:
        print("len(contours) == 0")
        return None

    g_shape = mask.shape
    img_gaussian = fc.get_gaussian_image(center = [g_shape[1]/2,g_shape[0]/2],
                        eigen_vector1 = [1,0],
                        eigen_vector2 = [0,1],
                        eigen_value1 = g_shape[0]*gaussian_factor,
                        eigen_value2 = g_shape[1]*gaussian_factor,
                        img_shape = g_shape)

    img_contours_prob = img_gaussian * (mask.astype(np.uint32) / 255)
    total_gaussian_prob = np.sum(img_contours_prob)

    contours_prob = np.zeros((len(contours)))

    for i in range(len(contours)):
        contour_mask = fc.create_contours_mask(contours, mask.shape, i).astype(np.float32) / 255
        contours_prob[i] = np.sum(img_contours_prob * contour_mask)


    final_probs = contours_prob / total_gaussian_prob

    final_mask = np.zeros_like(mask)
    best_prob = np.max(final_probs)
    safe_prob_area = safp * (best_prob - np.min(final_probs))

    for i in range(len(contours)):
        if best_prob - final_probs[i] <= safe_prob_area:
            cv.drawContours(final_mask, contours, i, (255,255,255), -1)

    if resize_w is not None:
        final_mask = fc.horizontal_resize(final_mask, img_mask.shape[1], force_h=img_mask.shape[0], interpolation=cv.INTER_NEAREST)

    return final_mask


def refine(img_mask, circle_mask=None, medianb_size=7, gaussian_factor=6, safp=99/100, resize_w=512):
    if circle_mask is not None:
        mask = fc.apply_mask(img_mask, circle_mask, -1)
    else:
        mask = img_mask.copy()

    mask = remove_border_contours_and_fill_holes(mask, medianb_size, gaussian_factor, safp, resize_w)
    
    if mask is None:
        print("none")
        circle = fc.get_contained_circle(img_mask.shape, 0.8)
        mask = fc.create_circle_mask(circle, img_mask.shape)
        # if circle_mask is not None:
        #     return circle_mask
        # return np.ones_like(img_mask) * 255
    return mask