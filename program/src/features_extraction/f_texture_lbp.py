import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_del_paquete)

import cv2 as cv
# import numpy as np
# import functions.functions as fc
from . import util as feat_util


def _extract_texture(img_lbp, multiple_radius, num_point, mask=None):    
    return tuple(feat_util.extract_lbp_features_from_lbp_image(img_lbp, multiple_radius, num_point, mask))



def _extract_texture(img_lbp, multiple_radius, num_point, mask=None, spatial_info=False):
    if spatial_info:
        # obtaining grid division
        img_shape = img_lbp.shape[:2]
        i_roi = feat_util.get_target_roi(img_shape, mask)
        boundaries = feat_util.divide_roi(i_roi, 3, img_shape, 3, 3)
        all_feat = []

        # obtaining features
        for roi in boundaries:
            img_roi = img_lbp[roi[1]:roi[3], roi[0]:roi[2]]
            all_feat.extend(tuple(feat_util.extract_lbp_features_from_lbp_image(img_roi, multiple_radius, num_point, None)))

        return tuple(all_feat)
    else:
        return tuple(feat_util.extract_lbp_features_from_lbp_image(img_lbp, multiple_radius, num_point, mask))



def f_extract_texture_lbp(img_lbp, args_dict=None):
    return _extract_texture(img_lbp, args_dict["multiple_radius"], args_dict["num_point"])

def f_extract_texture_lbp_mask(img_lbp, args_dict=None):
    return _extract_texture(img_lbp, args_dict["multiple_radius"], args_dict["num_point"], args_dict["mask"])

def f_extract_texture_lbp_mask_spatial(img_lbp, args_dict=None):
    return _extract_texture(img_lbp, args_dict["multiple_radius"], args_dict["num_point"], args_dict["mask"], True)
