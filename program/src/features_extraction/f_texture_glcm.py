import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_del_paquete)

import cv2 as cv
# import numpy as np
# import functions.functions as fc
from . import util as feat_util


# def _extract_texture(img, mask=None):    
#     gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     return tuple(feat_util.extract_glcm(gray_image, mask))

def _extract_texture(img, mask=None, spatial_info=False, gray_labels=None):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if spatial_info:
        # obtaining grid division
        img_shape = img.shape[:2]
        i_roi = feat_util.get_target_roi(img_shape, mask)
        boundaries = feat_util.divide_roi(i_roi, 3, img_shape, 3, 3)
        all_feat = []

        # obtaining features
        for roi in boundaries:
            img_roi = img[roi[1]:roi[3], roi[0]:roi[2]]
            all_feat.extend(tuple(feat_util.extract_glcm(img_roi, None, gray_labels=gray_labels)))

        return tuple(all_feat)
    else:
        return tuple(feat_util.extract_glcm(img, mask, gray_labels=gray_labels))



def f_extract_texture_glcm(img, args_dict=None):
    return _extract_texture(img, gray_labels=args_dict["gray_labels"])

def f_extract_texture_glcm_mask(img, args_dict=None):
    return _extract_texture(img, args_dict["mask"], gray_labels=args_dict["gray_labels"])

def f_extract_texture_glcm_mask_spatial(img, args_dict=None):
    return _extract_texture(img, args_dict["mask"], spatial_info=True, gray_labels=args_dict["gray_labels"])