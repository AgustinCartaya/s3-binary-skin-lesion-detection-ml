import cv2 as cv
import numpy as np

import functions.functions as fc
from . import prob_labels_filters as prob


def get_gray_labels(img_labels_ems, add_bg=True):
    img_labels_ems_gray = cv.cvtColor(img_labels_ems, cv.COLOR_BGR2GRAY)

    if add_bg:
        unique_values = np.unique(img_labels_ems_gray)
        for i in range(len(unique_values)):
            img_labels_ems_gray[img_labels_ems_gray == unique_values[i]] = i+1
        img_labels_ems_gray = (255 *  img_labels_ems_gray.astype(np.float32)/ (len(unique_values))).astype(np.uint8)
    return img_labels_ems_gray


def get_best_prob_mask_V1(img_prep, img_labels_ems, circular_mask):
    """Image no resized !! """
    # convert img_labels_ems from color to gray scale to obtain the possible labels
    img_labels_ems_gray = get_gray_labels(img_labels_ems, add_bg=True)

    # obtain possible labels (max = 2*3*4 = 24)
    lbs = np.unique(img_labels_ems_gray)
    n_lbs = len(lbs)

    # create first probabilistic matrix and weights (importance)
    probs = np.zeros((5, n_lbs), dtype=np.float64)
    importance = np.array([2,0.5,0.5,1,3])

    # create circles for center tests
    c1_circle = fc.get_img_centered_circle(img_labels_ems_gray.shape, 0.1)
    c1_mask = fc.create_circle_mask(c1_circle, img_labels_ems_gray.shape)
    c2_circle = fc.get_img_centered_circle(img_labels_ems_gray.shape, 0.3)
    c2_mask = fc.create_circle_mask(c2_circle, img_labels_ems_gray.shape)

    #---------- first probabilities
    probs[0, :] = prob.get_prob_outher_circle(img_labels_ems_gray, circular_mask, lbs) 
    probs[1, :] = prob.get_prob_inside_mask(img_labels_ems_gray, c1_mask, lbs)
    probs[2, :] = prob.get_prob_inside_mask(img_labels_ems_gray, c2_mask, lbs)
    probs[3, :] = prob.get_prob_IoU_mask(img_labels_ems_gray, c2_mask, lbs)
    probs[4, :] = prob.get_prob_most_different_mean(img_labels_ems_gray, img_prep, lbs, channels=[0,1,2])

    # compute first final prob 
    _, ordered_contorus, _ = prob.compute_final_prob(probs, importance, lbs)

    # create first mask with best contour
    first_mask = np.zeros_like(img_labels_ems_gray)
    first_mask[(img_labels_ems_gray == ordered_contorus[0])] = 255
    # fc.imgshow("first_mask" , first_mask, 0.5)

    #---------- second probabilities
    prob_color_mean_lab = prob.get_prob_similar_mean_to_mask(img_labels_ems_gray, cv.cvtColor(img_prep, cv.COLOR_BGR2Lab), ordered_contorus[0], lbs, channels=[0,1,2])
    probs = np.vstack((probs, prob_color_mean_lab))
    importance = np.hstack((importance, 6))

    final_probs, ordered_contorus, ordered_probs = prob.compute_final_prob(probs, importance, lbs)

    # add second probabilities to the first mask
    std_factor = 1.2
    std_final_probs = std_factor * np.std(final_probs)
    for i in range(1, n_lbs):
        diff = (ordered_probs[0] - ordered_probs[i])
        if diff < std_final_probs:
            first_mask[(img_labels_ems_gray == ordered_contorus[i])] = 255

    #---------- select most probable segmentation between em_k=2,3,4
    final_mask = prob.get_prob_IoU_em_234(img_labels_ems, first_mask, ths=[0.3, 0.5, 0.7])
    if final_mask is None:
        final_mask = first_mask

    return final_mask



def get_best_prob_mask_V2(img_prep, img_labels_ems, outer_circle):
    """Image no resized !! """

    # convert img_labels_ems from color to gray scale to obtain the possible labels
    img_prep_resized = fc.horizontal_resize(img_prep, 512, interpolation=cv.INTER_NEAREST)
    img_labels_ems_resized = fc.horizontal_resize(img_labels_ems, 512, interpolation=cv.INTER_NEAREST)
    outer_circle = fc.horizontal_resize(outer_circle, 512, interpolation=cv.INTER_NEAREST)

    img_labels_ems_gray = get_gray_labels(img_labels_ems_resized, add_bg=True)
    img_labels_ems_gray = cv.medianBlur(img_labels_ems_gray, 11)

    # obtain possible labels (max = 2*3*4 = 24)
    lbs = np.unique(img_labels_ems_gray)
    n_lbs = len(lbs)

    probs = np.zeros((4, n_lbs), dtype=np.float64)
    importance = np.array([2,1,1,4])

    # create circles for center tests
    c2_circle = fc.get_contained_circle(img_labels_ems_gray.shape, 0.3)
    c2_mask = fc.create_circle_mask(c2_circle, img_labels_ems_gray.shape)

    #---------- first probabilities
    probs[0, :] = prob.get_prob_outher_circle(img_labels_ems_gray, outer_circle, lbs) 
    probs[1, :] = prob.get_prob_inside_gaussian(img_labels_ems_gray, lbs)
    probs[2, :] = prob.get_prob_IoU_mask(img_labels_ems_gray, c2_mask, lbs)
    probs[3, :] = prob.get_prob_most_different_mean(img_labels_ems_gray, img_prep_resized, lbs, channels=[0,1,2])

    # compute first final prob 
    _, ordered_contorus, _ = prob.compute_final_prob(probs, importance, lbs)

    # create first mask with best contour
    first_mask = np.zeros_like(img_labels_ems_gray)
    first_mask[(img_labels_ems_gray == ordered_contorus[0])] = 255
    # fc.imgshow("first_mask", first_mask, 0.5)

    #---------- second probabilities
    prob_color_mean_lab = prob.get_prob_similar_mean_to_mask(img_labels_ems_gray, cv.cvtColor(img_prep_resized, cv.COLOR_BGR2Lab), ordered_contorus[0], lbs, channels=[0,1,2])
    probs = np.vstack((probs, prob_color_mean_lab))
    importance = np.hstack((importance, 6))

    final_probs, ordered_contorus, ordered_probs = prob.compute_final_prob(probs, importance, lbs)

    # add second probabilities to the first mask
    std_factor = 1.2
    std_final_probs = std_factor * np.std(final_probs)

    for i in range(1, n_lbs):
        diff = (ordered_probs[0] - ordered_probs[i])
        if diff < std_final_probs:
            first_mask[(img_labels_ems_gray == ordered_contorus[i])] = 255

    #---------- select most probable segmentation between em_k=2,3,4
    final_mask = prob.get_prob_IoU_em_234(img_labels_ems_resized, first_mask, ths=[0.3, 0.5, 0.7])
    if final_mask is None:
        final_mask = first_mask


    final_mask = fc.horizontal_resize(final_mask, img_prep.shape[1], force_h=img_prep.shape[0], interpolation=cv.INTER_NEAREST)

    return final_mask