import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_del_paquete)

import skin_functions.preprocessing.outer_circle_remover as outer_circle_remover
# import numpy as np
import cv2 as cv
import functions.functions as fc

POSTPROCESSING_V1 = "postprocessing_v1"

def get_reduced_roi_if_outher_circle(img_original):
    outher_circle = outer_circle_remover.get_outer_circle(img_original)
    if outher_circle is not None:
        roi_w = int(outher_circle[0][1] / (2 ** 0.5))
        x1_rec, y1_rec, x2_rec, y2_rec = outher_circle[1]
        img_shape = y2_rec-y1_rec, x2_rec-x1_rec
        img_center_y, img_center_x = img_shape[0]//2, img_shape[1]//2
        roi = max(0, img_center_x - roi_w), max(0, img_center_y - roi_w), min(img_shape[1], img_center_x + roi_w), min(img_shape[0], img_center_y + roi_w)
        return roi
    return None


def reduce_excesive_mask(mask, img_shape, max_factor, r_factor):
    if cv.countNonZero(mask) > (img_shape[0] * img_shape[1] * max_factor):
        reduced_circle = fc.get_contained_circle(img_shape, r_factor)
        return fc.create_circle_mask(reduced_circle, img_shape)
    return mask


def postprocess1(img_original, img_prep, mask=None, mask_max_factor=0.9, mask_reduction_factor=0.7):
    # ---- remove outhercircle if contains and reduce exesive masks
    reduced_roi = get_reduced_roi_if_outher_circle(img_original)
    if reduced_roi is not None:   
        img_prep = fc.get_roi(img_prep, reduced_roi)
        if mask is not None:
            mask = fc.get_roi(mask, reduced_roi)
        
    # ---- reduce excesive mask
    if mask is not None:
        mask = reduce_excesive_mask(mask, img_prep.shape[:2], max_factor=mask_max_factor, r_factor=mask_reduction_factor)
        return img_prep, mask

    return img_prep