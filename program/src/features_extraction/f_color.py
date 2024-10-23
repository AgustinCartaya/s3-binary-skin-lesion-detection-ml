
import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_del_paquete)

# import cv2 as cv
# import numpy as np
# import functions.functions as fc
from . import util as feat_util


def _extract_color(img, color_space, mask=None, spatial_info=False):
    if spatial_info:
        # obtaining grid division
        img_shape = img.shape[:2]
        i_roi = feat_util.get_target_roi(img_shape, mask)
        boundaries = feat_util.divide_roi(i_roi, 3, img_shape, 3, 3)
        all_feat = []

        # obtaining features
        for roi in boundaries:
            img_roi = img[roi[1]:roi[3], roi[0]:roi[2]]
            all_feat.extend(tuple(feat_util.extract_color_features(img_roi, color_space, None)))

        return tuple(all_feat)
    else:
        return tuple(feat_util.extract_color_features(img, color_space, mask))


def f_extract_color(img, args_dict=None):
    return _extract_color(img, "bgr") + _extract_color(img, "hls") + _extract_color(img, "hsv") + _extract_color(img, "lab")

def f_extract_color_mask(img, args_dict=None):
    return _extract_color(img, "bgr", args_dict["mask"]) + _extract_color(img, "hls", args_dict["mask"]) + _extract_color(img, "hsv", args_dict["mask"]) + _extract_color(img, "lab", args_dict["mask"])

def f_extract_color_mask_spatial(img, args_dict=None):
    return _extract_color(img, "bgr", args_dict["mask"], True) + _extract_color(img, "hls", args_dict["mask"], True) + _extract_color(img, "hsv", args_dict["mask"], True) + _extract_color(img, "lab", args_dict["mask"], True)



