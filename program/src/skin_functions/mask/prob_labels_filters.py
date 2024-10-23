import cv2 as cv
import numpy as np

import functions.functions as fc


def get_prob_outher_circle(img_labels_ems_gray, circular_mask, lbs):
    n_lbs = len(lbs)
    probs = np.ones((n_lbs)) / n_lbs

    if circular_mask is not None:
        img_labels_ems_gray_maked = fc.apply_mask(img_labels_ems_gray, cv.bitwise_not(circular_mask), mask_value=-1)
        unique_values, counts = np.unique(img_labels_ems_gray_maked, return_counts=True)

        bg_area = np.count_nonzero(circular_mask == 0)

        for i in range(n_lbs):
            index = np.where(unique_values == lbs[i])[0]
            if len(index) > 0:
                index = index[0]
                probs[i] = counts[index] / bg_area

        probs = 1 - probs
        probs = probs / np.sum(probs)
    return probs


def get_prob_inside_mask(img_labels_ems_gray, mask, lbs):
    n_lbs = len(lbs)
    probs = np.zeros((n_lbs))

    img_labels_ems_gray_maked = fc.apply_mask(img_labels_ems_gray, mask, mask_value=-1)
    unique_values, counts = np.unique(img_labels_ems_gray_maked, return_counts=True)

    circle_area = np.count_nonzero(mask > 0)
    for i in range(n_lbs):
        index = np.where(unique_values == lbs[i])[0]
        if len(index) > 0:
            index = index[0]
            probs[i] = counts[index] / circle_area

    return probs


def get_prob_inside_gaussian(img_labels_ems_gray, lbs, gaussian_factor=10):
    n_lbs = len(lbs)
    probs = np.zeros((n_lbs))

    g_shape = img_labels_ems_gray.shape
    img_gaussian = fc.get_gaussian_image(center = [g_shape[1]/2,g_shape[0]/2],
                        eigen_vector1 = [1,0],
                        eigen_vector2 = [0,1],
                        eigen_value1 = g_shape[0]*gaussian_factor,
                        eigen_value2 = g_shape[1]*gaussian_factor,
                        img_shape = g_shape)

    total_gaussian_prob = np.sum(img_gaussian)

    for i in range(n_lbs):
        img_lb_mask = np.zeros_like(img_labels_ems_gray, dtype=np.float32)
        img_lb_mask[img_labels_ems_gray == lbs[i]] = 1
        probs[0] = np.sum(img_gaussian * img_lb_mask)

    probs = probs/total_gaussian_prob
    return probs


def get_prob_IoU_mask(img_labels_ems_gray, mask, lbs):
    n_lbs = len(lbs)
    probs = np.zeros((n_lbs))

    for i in range(n_lbs):
        lb_i_mask = np.zeros_like(img_labels_ems_gray)
        lb_i_mask[img_labels_ems_gray == lbs[i]] = 255

        intersection = cv.countNonZero(cv.bitwise_and(lb_i_mask, mask))
        union = cv.countNonZero(cv.bitwise_or(lb_i_mask, mask))
        iou = intersection / union

        probs[i] = iou
 
    probs = probs / np.sum(probs)
    return probs


# def get_prob_most_different_mean(img_labels_ems_gray, img_prep, lbs, channels=[0,1,2]):
#     n_lbs = len(lbs)
#     probs = np.zeros((n_lbs))

#     chn_list = []
#     for i in channels:
#         chn_list.append(np.mean(img_prep[:,:,i]))
#     global_centroid = np.array(chn_list)

#     for i in range(n_lbs):
#         _chn_list = []
#         for channel in channels:
#             _chn_list.append(np.mean(img_prep[:,:,channel][img_labels_ems_gray == lbs[i]]))
#         label_centroid = np.array(_chn_list)
#         distace = np.linalg.norm(global_centroid - label_centroid)
#         probs[i] = distace/np.sqrt(len(channels)*255**2)

#     probs = probs / np.sum(probs)
#     return probs


def get_prob_most_different_mean(img_labels_ems_gray, img_prep, lbs, channels=[0,1,2]):
    n_lbs = len(lbs)
    probs = np.zeros((n_lbs))

    img_prep_flat = img_prep.transpose(2, 0, 1).reshape(3, -1)
    global_mean = np.mean(img_prep_flat[channels], axis=1) 

    img_labels_ems_gray_flat = img_labels_ems_gray.ravel()
    for i in range(n_lbs):
        img_prep_lb = img_prep_flat[channels][:, img_labels_ems_gray_flat == lbs[i]]
        lb_mean = np.mean(img_prep_lb, axis=1) 

        distace = np.linalg.norm(global_mean - lb_mean)
        probs[i] = distace/np.sqrt(len(channels)*255**2)

    probs = probs / np.sum(probs)
    return probs


def get_prob_similar_mean_to_mask(img_labels_ems_gray, img_prep, best_label, lbs, channels=[0,1,2]):
    n_lbs = len(lbs)
    probs = np.zeros((n_lbs))

    chn_list = []
    for channel in channels:
        chn_list.append(np.mean(img_prep[:,:,channel][img_labels_ems_gray == best_label]))
    color_centroid = np.array(chn_list)

    for i in range(n_lbs):
        if lbs[i] == best_label:
            continue

        _chn_list = []
        for channel in channels:
            _chn_list.append(np.mean(img_prep[:,:,channel][img_labels_ems_gray == lbs[i]]))
        _color_centroid = np.array(_chn_list)

        distace = np.linalg.norm(color_centroid - _color_centroid)
        probs[i] = distace/np.sqrt(len(channels)*255**2)

    probs = 1-probs
    probs = probs / np.sum(probs)
    return probs
    
 
def get_prob_IoU_em_234(img_labels_ems, mask, ths= [0.6, 0.6, 0.9]):

    # loop al the em k
    for em_k in range(img_labels_ems.shape[2]):
        img_em_k = img_labels_ems[:,:,em_k]
        labels_em_k = np.unique(img_em_k) 
        # loop all the labels
        for label in labels_em_k:
            # create label mask
            mask_label = np.zeros_like(img_em_k)
            mask_label[img_em_k == label] = 255

            intersection = cv.countNonZero(cv.bitwise_and(mask_label, mask))
            union = cv.countNonZero(cv.bitwise_or(mask_label, mask))
            iou = intersection / union

            if iou >= ths[em_k]:
                mask_label[mask_label > 0] = 255
                mask_label[mask > 0] = 255
                return mask_label
    return None


def compute_final_prob(probs, importance, lbs):
    final_probs = np.sum(probs * importance[:, np.newaxis], axis=0) / np.sum(importance)
    index_dsc = np.argsort(final_probs)[::-1]
    ordered_contorus = [lbs[i] for i in  index_dsc]
    ordered_probs = [final_probs[i] for i in  index_dsc]

    return final_probs, ordered_contorus, ordered_probs