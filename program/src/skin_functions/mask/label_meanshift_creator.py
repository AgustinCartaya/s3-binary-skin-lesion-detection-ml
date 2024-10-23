import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_del_paquete)

import cv2 as cv
import functions.functions as fc


def get_labels(img_prep, spatial_radius=10, color_radius=20, resize_w=256):
    _img_prep_resized = fc.horizontal_resize(img_prep, resize_w)
    labels = cv.pyrMeanShiftFiltering(_img_prep_resized, spatial_radius, color_radius, maxLevel=2)
    labels = fc.horizontal_resize(labels, img_prep.shape[1], force_h=img_prep.shape[0], interpolation=cv.INTER_NEAREST)
    return labels

