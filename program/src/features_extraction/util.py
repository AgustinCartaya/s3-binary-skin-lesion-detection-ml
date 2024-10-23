import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_del_paquete)

import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

import functions.functions as fc


# -------- SPATIAL DIVISION
def get_target_roi(img_shape, mask=None):
    if mask is not None:
        contour = fc.get_contours(mask, order_by_area=True)[0]
        x, y, w, h = cv.boundingRect(contour)
        i_roi = x, y, x+w, y+h
    else:
        i_roi = 0,0, img_shape[1], img_shape[0]

    return i_roi

        
def expand_bounding_rec(x1, y1, x2, y2, img_shape, min_w, min_h):
    if ((x2-x1) >= min_w and (y2-y1) >= min_h) or (x1==0 and x2==img_shape[1]) or (y1==0 and y2==img_shape[0]):
        return x1, y1, x2, y2

    if (x2-x1) < min_w:
        return expand_bounding_rec(max(x1-1,0), y1, min(x2+1, img_shape[1]), y2, img_shape, min_w, min_h)

    if (y2-y1) < min_h:
        return expand_bounding_rec(x1, max(y1-1,0), x2, min(y2+1, img_shape[0]), img_shape, min_w, min_h)


def divide_number_parts(a, n):
    partes = [a // n] * n  

    residuo = a % n
    for i in range(residuo):
        partes[i] += 1

    return partes


def divide_roi(roi, divisions, img_shape, min_division_w, min_division_h):
    # expand bounding rect if too small
    min_w=min_division_w*divisions
    min_h=min_division_h*divisions
    expanded_roi = expand_bounding_rec(roi[0], roi[1], roi[2], roi[3], img_shape=img_shape[:2], min_w=min_w, min_h=min_h)

    # obtain divisions
    x_division = np.array([0]+divide_number_parts(expanded_roi[2] - expanded_roi[0], divisions))
    y_division = np.array([0]+divide_number_parts(expanded_roi[3] - expanded_roi[1], divisions))

    x_boundaries = np.cumsum(x_division) + expanded_roi[0]
    y_boundaries = np.cumsum(y_division) + expanded_roi[1]

    boundaries = np.zeros((divisions*divisions, 4), dtype=np.int16)

    counter = 0
    for x in range(len(x_boundaries)-1):
        for y in range(len(x_boundaries)-1):
            boundaries[counter] = (x_boundaries[x], y_boundaries[y], x_boundaries[x+1], y_boundaries[y+1])
            counter +=1

    return boundaries
            
        


# -------- COLOR
def extract_color_features(img, color_space, mask=None):
    if color_space == "bgr":
        w_img = img
    elif color_space == "hls":
        w_img = cv.cvtColor(img, cv.COLOR_BGR2HLS)

    elif color_space == "hsv":
        w_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    elif color_space == "lab":
        w_img = cv.cvtColor(img, cv.COLOR_BGR2Lab)


    if mask is not None:
        chnn_1, chnn_2, chnn_3 = fc.get_values_in_mask(w_img, mask)
    else:
        chnn_1, chnn_2, chnn_3 = cv.split(w_img)

    mean_chnn_1 = np.mean(chnn_1)
    mean_chnn_2 = np.mean(chnn_2)
    mean_chnn_3 = np.mean(chnn_3)

    sd_chnn_1 = np.std(chnn_1)
    sd_chnn_2 = np.std(chnn_2)
    sd_chnn_3 = np.std(chnn_3)

    dominance = max(mean_chnn_1, mean_chnn_2, mean_chnn_3)
    contrast = (sd_chnn_1 + sd_chnn_2 + sd_chnn_3) / 3.0
    coherence = 1.0 - (contrast / dominance)

    return np.array([mean_chnn_1, mean_chnn_2, mean_chnn_3, sd_chnn_1, sd_chnn_2, sd_chnn_3, dominance, contrast, coherence])


# -------- LBP
def extract_lbp_features_from_lbp_image(img_lbp, multiple_radius=[1,2,3], num_point=8, mask=None):
    complet_hist = np.zeros(np.sum(np.array(multiple_radius)*num_point)+2*len(multiple_radius))
    old_p = 0
    for i in range(len(multiple_radius)):
        p =  num_point * multiple_radius[i]
        lbp = img_lbp[:,:,i]

        if mask is not None:
            lbp = fc.apply_mask(lbp, cv.bitwise_not(mask), -2)

        hist, _ = fc.calc_hist(lbp, normalize=True, range=(0,p+2), num_bins=p+2, norm="l2")
   
        complet_hist[old_p:old_p+p+2] = hist
        old_p = old_p+p+2
    return complet_hist




# -------- GLCM
def calculate_haralick_features(glcm):
    haralick_features = graycoprops(glcm)
    haralick_mean = np.mean(haralick_features, axis=0)
    return haralick_mean

def calculate_cluster_prominence(glcm):
    p = np.indices(glcm.shape)[0]
    q = np.indices(glcm.shape)[1]

    mean_row = np.sum(p * glcm) / np.sum(glcm)
    mean_col = np.sum(q * glcm) / np.sum(glcm)

    cluster_prominence = np.sum(((p + q - mean_row - mean_col) ** 4) * glcm) / np.sum(glcm) ** 2
    return cluster_prominence

def calculate_cluster_shade(glcm):
    p = np.indices(glcm.shape)[0]
    q = np.indices(glcm.shape)[1]
    mean_row = np.sum(p * glcm) / np.sum(glcm)
    mean_col = np.sum(q * glcm) / np.sum(glcm)
    cluster_shade = np.sum(((p + q - mean_row - mean_col) ** 3) * glcm) / np.sum(glcm) ** 2
    return cluster_shade

def calculate_max_probability(glcm):
    max_probability = np.max(glcm) / np.sum(glcm)
    return max_probability

def calculate_sum_average(glcm):
    p = np.indices(glcm.shape)[0]
    q = np.indices(glcm.shape)[1]
    sum_average = np.sum((p + q) * glcm) / np.sum(glcm)
    return sum_average

def calculate_sum_variance(glcm):
    p = np.indices(glcm.shape)[0]
    q = np.indices(glcm.shape)[1]
    sum_average = calculate_sum_average(glcm)
    sum_variance = np.sum(((p + q) - sum_average) ** 2 * glcm) / np.sum(glcm)
    return sum_variance

def calculate_sum_entropy(glcm):
    glcm_normalized = glcm / np.sum(glcm)
    sum_entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))
    return sum_entropy

def calculate_difference_variance(glcm):
    p = np.indices(glcm.shape)[0]
    q = np.indices(glcm.shape)[1]
    difference_variance = np.sum(((p - q) ** 2) * glcm) / np.sum(glcm)
    return difference_variance

def calculate_difference_entropy(glcm):
    glcm_normalized = glcm / np.sum(glcm)
    difference_entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))
    return difference_entropy


def extract_glcm(img, mask=None, gray_labels=None):
    if mask is not None:
        img = fc.apply_mask(img, mask, 255)

    if gray_labels is not None:
        img = (img.astype(np.float16) * (gray_labels - 1) / 255).astype(np.uint8)
        glcm = graycomatrix(img, distances=[3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=gray_labels)
    else:
        glcm = graycomatrix(img, distances=[3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])


    if mask is not None:
        if gray_labels is None:
            glcm[255, 255, :, :] = 0
        else:
            glcm[gray_labels, gray_labels, :, :] = 0

    # print(np.sum(glcm))
    # print((img.shape[0] * (img.shape[1]-1))-np.sum(mask/255))
    # glcm = graycomatrix(roi, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    asm = graycoprops(glcm, 'ASM')

    cluster_prominence = calculate_cluster_prominence(glcm)
    cluster_shade = calculate_cluster_shade(glcm)
    max_probability = calculate_max_probability(glcm)
    sum_average = calculate_sum_average(glcm)
    sum_variance = calculate_sum_variance(glcm)
    sum_entropy = calculate_sum_entropy(glcm)
    difference_variance = calculate_difference_variance(glcm)
    difference_entropy = calculate_difference_entropy(glcm)
    haralick = calculate_haralick_features(glcm)

    lacunarity = 1 - (contrast.var() / contrast.mean() ** 2)

    features = np.concatenate([
        contrast.ravel(),
        dissimilarity.ravel(),
        homogeneity.ravel(),
        energy.ravel(),
        correlation.ravel(),
        asm.ravel(),
        haralick.ravel(),
        lacunarity.ravel(),             # number
        cluster_prominence.ravel(),     # number
        cluster_shade.ravel(),          # number
        max_probability.ravel(),        # number
        sum_average.ravel(),            # number
        sum_variance.ravel(),           # number
        sum_entropy.ravel(),            # number
        difference_variance.ravel(),    # number
        difference_entropy.ravel()      # number
    ])

    return features
