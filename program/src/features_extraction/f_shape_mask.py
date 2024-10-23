import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_del_paquete)

import cv2 as cv
import numpy as np
import functions.functions as fc
from . import util as feat_util



def _extract_shape(img, mask):
    contours = fc.get_contours(mask, order_by_area=True)
    contour = contours[0]

    # mask sharp
    points = contour[:, 0, :]  
    angles = np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0]))
    angle_diffs = np.abs(np.diff(angles))
    threshold_angle_diff = np.pi / 3  
    total_diff = np.sum(angle_diffs > threshold_angle_diff)

    # circularity
    perimeter = cv.arcLength(contour, closed=True)  
    area = cv.contourArea(contour)
    compacity = (4 * np.pi * area) / (perimeter ** 2)

    # blur
    _,_,w,h = cv.boundingRect(contour)
    centroid = fc.get_contour_centroid(contour)
    init_points = fc.get_circle_points(mask.shape, 20, 30, centroid)
    end_points = fc.get_circle_points(mask.shape, 20, np.sqrt((w/2)**2 + (h/2)**2), centroid)

    _img_variances = np.zeros((init_points.shape[0]))
    for i in range(init_points.shape[0]):              
        p_init = int(img[init_points[i,1], init_points[i,0], 0])
        p_fin = int(img[end_points[i,1], end_points[i,0], 0])
        _img_variances[i] = np.abs( p_init- p_fin)

    _img_variances = np.sort(_img_variances)
    blur = np.percentile(_img_variances, 60)

    return total_diff, compacity, perimeter/area, blur



def f_extract_shape(img, args_dict=None):
    return _extract_shape(img, args_dict["mask"])


