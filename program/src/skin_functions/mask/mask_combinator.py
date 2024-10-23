import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_del_paquete)

import cv2 as cv
import numpy as np
import functions.functions as fc


def combine_masks(masks, img_prep):
    num_mask = len(masks)
    combination_list = fc.get_combination_list(num_mask)
    
    img_r = img_prep[:,:,2]
    img_r = cv.medianBlur(img_r, 15)

    base_std = np.std(img_r)

    nb_combinations = len(combination_list)
    probs = np.zeros((3,nb_combinations))
    
    # interception percentage
    interception_percent = np.zeros((num_mask))
    for i in range(num_mask):
        for j in range(num_mask):
            if i==j:
                continue
            intersection = cv.countNonZero(cv.bitwise_and(masks[i], masks[j]))
            union = cv.countNonZero(cv.bitwise_or(masks[i], masks[j]))
            iou = intersection / union
            interception_percent[i] += iou
    interception_percent /= np.sum(interception_percent)  

    for i in range(nb_combinations):
        _mask =  np.zeros_like(masks[0])

        for mask_index in combination_list[i]:
            _mask = cv.bitwise_or(_mask, masks[mask_index])
            probs[2, i] += interception_percent[mask_index]

        probs[2, i] /= len(combination_list[i])

        if np.any(_mask == 0):
            probs[0, i] = np.std(img_r[_mask==0])
        else:
            probs[0, i] = base_std

        probs[1, i] = cv.countNonZero(_mask)

    # prob std
    probs[0] = np.abs(base_std - probs[0])
    probs[0] = probs[0] / np.sum(probs[0])

    # prob area
    min_area = np.min(probs[1])
    probs[1] = np.abs(min_area - probs[1])
    probs[1] = np.exp( (1 - (probs[1] / np.sum(probs[1])))*3)
    probs[1] = probs[1] / np.sum(probs[1])

    # final probs
    importance = np.array([1,1,1])
    final_probs = np.sum(probs * importance[:, np.newaxis], axis=0) / np.sum(importance)

    # final mask
    final_mask =  np.zeros_like(masks[0])
    for mask_index in combination_list[np.argmax(final_probs)]:
        final_mask = cv.bitwise_or(final_mask, masks[mask_index])

    return final_mask



