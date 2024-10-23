import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_del_paquete)


import cv2 as cv
# import numpy as np

import functions.functions as fc



def get_border_mask(mask, inside_window=7, outside_window=15):
    mask_bigger_contour_erode = fc.fast_morph(mask, "erode", kernel_size=inside_window, w_size=256)
    mask_bigger_contour_dilate = fc.fast_morph(mask, "dilate", kernel_size=outside_window, w_size=256)
    mask_border = cv.bitwise_xor(mask_bigger_contour_erode, mask_bigger_contour_dilate)
    return mask_border




