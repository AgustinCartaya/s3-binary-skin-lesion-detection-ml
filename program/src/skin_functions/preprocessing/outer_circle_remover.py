import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_del_paquete)

import numpy as np
import cv2 as cv
import functions.functions as fc


def get_skin_hist_peak(img, th=128):
    """
    Search for the most common intensity after the 'th' in the histogram of the image.
    """
    hist, bins = fc.calc_hist(img=img, range=[0, 255], num_bins=255)
    rigth_hist = hist[th:]
    peak = th + np.argmax(rigth_hist)
    
    return bins[peak]


def get_outer_circle(img, reduction_circel=0.1, window_size=7, min_corner_th=40):
    """
    Return the circle that encloses the image without background and the region of interest
    """
    img_red = img[:,:,2]
    img_h, img_w = img_red.shape[:2]

    if (np.max(img_red[:window_size, :window_size]) < min_corner_th or
        np.max(img_red[img_h - window_size:, :window_size]) < min_corner_th or
        np.max(img_red[:window_size, img_w - window_size:]) < min_corner_th or
        np.max(img_red[img_h - window_size:, img_w - window_size:]) < min_corner_th):

        _, initial_mask = cv.threshold(img_red,min_corner_th,255,cv.THRESH_BINARY)
        contours = fc.get_contours(initial_mask, order_by_area=True)

        # obtaining outher circle
        (x_circle, y_circle), radius = cv.minEnclosingCircle(contours[0])
        center = (int(x_circle), int(y_circle))
        reduced_radius = int(radius * (1-reduction_circel))

        # obtaining roi
        x1_rec, y1_rec, x2_rec, y2_rec = (max(0,int(x_circle - reduced_radius)), 
                                         max(0,int(y_circle - reduced_radius)), 
                                         min(int(x_circle + reduced_radius), img_w),
                                         min(int(y_circle + reduced_radius), img_h))

        return (center, reduced_radius), (x1_rec, y1_rec, x2_rec, y2_rec)
    return None


def get_preprocessed_circular_mask(img, reduction_circel=0.1, horizontal_resize=None):
    outer_circle = get_outer_circle(img, reduction_circel)
    if outer_circle is not None:
        circle, roi = outer_circle
        x1_rec, y1_rec, x2_rec, y2_rec = roi

        circle_translated = ((circle[0][0]-x1_rec, circle[0][1]-y1_rec), circle[1])

        img_no_corner = img[y1_rec:y2_rec, x1_rec:x2_rec]
        ch,cw = img_no_corner.shape[:2]
    
        circular_mask = fc.create_circle_mask(circle_translated, (ch,cw))
        if horizontal_resize is not None:
            circular_mask = fc.horizontal_resize(circular_mask, horizontal_resize, interpolation=cv.INTER_NEAREST)
        return circular_mask
    return None


def remove_outer_circle(img, new_bg_color=None, mean_bg_th = 150):
    outer_circle = get_outer_circle(img)

    if outer_circle is not None:
        circle, roi = outer_circle
        x1_rec, y1_rec, x2_rec, y2_rec = roi

        circle_translated = ((circle[0][0]-x1_rec, circle[0][1]-y1_rec), circle[1])
        img_no_corner = img[y1_rec:y2_rec, x1_rec:x2_rec]
        ch,cw = img_no_corner.shape[:2]
        circular_mask = fc.create_circle_mask(circle_translated, (ch,cw))

        if new_bg_color is None:
            peak_b = get_skin_hist_peak(img_no_corner[:, :, 0], th=mean_bg_th)
            peak_g = get_skin_hist_peak(img_no_corner[:, :, 1], th=mean_bg_th)
            peak_r = get_skin_hist_peak(img_no_corner[:, :, 2], th=mean_bg_th)
            img_no_corner = fc.apply_mask(img_no_corner, cv.bitwise_not(circular_mask), (peak_b, peak_g, peak_r)) 
        else:
            img_no_corner = fc.apply_mask(img_no_corner, cv.bitwise_not(circular_mask), new_bg_color) 

        return img_no_corner
    return img
