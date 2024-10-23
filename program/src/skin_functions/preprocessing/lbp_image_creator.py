
from skimage.feature import local_binary_pattern
import numpy as np


def get_lbp_img(img, radius, num_point, method='uniform'):
    lbp_img = local_binary_pattern(img, P=num_point, R=radius, method=method)
    return lbp_img


def get_multiple_lbp_img(img, multiple_radius, num_points, method='uniform'):
    chn = len(multiple_radius)
    lbp_img = np.zeros((img.shape[0], img.shape[1], chn))
    for i in range(chn):
        p =  num_points * multiple_radius[i]
        lbp_img[:,:,i] = get_lbp_img(img, multiple_radius[i], p, method=method)
    return lbp_img