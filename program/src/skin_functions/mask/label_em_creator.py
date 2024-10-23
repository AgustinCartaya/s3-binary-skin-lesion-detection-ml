import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_del_paquete)

import numpy as np
import functions.functions as fc
import cv2 as cv


def get_labels(img, em_ks, resize_w=256, initial_interpolation=cv.INTER_NEAREST):
    nb_em = len(em_ks)
    _img_resized = fc.horizontal_resize(img, resize_w, interpolation=initial_interpolation)
    working_h, working_w = _img_resized.shape[:2]
    normalize_em = lambda img_em, nk: (255 *  img_em/(nk-1)).astype(np.uint8)
    imgs_em = [_img_resized[:,:,0], _img_resized[:,:,1], _img_resized[:,:,2]]

    if nb_em > 1:
        labels = np.zeros((working_h, working_w, 3), dtype=np.uint8)
        for i in range(nb_em):
            labels[:,:,i] = normalize_em(fc.expectation_maximization(imgs_em, em_ks[i]), em_ks[i])
    else:
        labels = normalize_em(fc.expectation_maximization(imgs_em, em_ks[0]), em_ks[0])

    labels = fc.horizontal_resize(labels, img.shape[1], force_h=img.shape[0], interpolation=cv.INTER_NEAREST)
    return labels
