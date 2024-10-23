import numpy as np
import cv2 as cv
import pathlib
import os
from sklearn.metrics import accuracy_score


# preprocessing
from .src.skin_functions.preprocessing import outer_circle_remover as outer_circle_remover
from .src.skin_functions.preprocessing import hair_remover as hair_remover
from .src.skin_functions.preprocessing import lbp_image_creator as lbp_image_creator
from .src.skin_functions.postprocessing import postprocessing as postprocessing


# mask
from .src.skin_functions.mask import mask_watershed_creator as mask_watershed_creator
from .src.skin_functions.mask import refiner as refiner
from .src.skin_functions.mask import label_meanshift_creator as label_meanshift_creator
from .src.skin_functions.mask import label_em_creator as label_em_creator
from .src.skin_functions.mask import mask_from_em234_labels_creator as mask_from_em234_labels_creator
from .src.skin_functions.mask import mask_combinator as mask_combinator
from .src.skin_functions.mask import mask_border_creator as mask_border_creator


# features extraction
from .src.features_extraction import f_color as f_color
from .src.features_extraction import f_texture_lbp as f_texture_lbp
from .src.features_extraction import f_texture_glcm as f_texture_glcm
from .src.features_extraction import f_shape_mask as f_shape_mask


# classification
from .src.classification.classifier import Classifier


# functions
from .src.functions import functions as fc

def get_base_folder():
    return str(pathlib.Path(__file__).parent.resolve()).replace(os.sep, "/") + "/"

def jpg_compresion(img):
    _, buffer = cv.imencode('.jpg', img)
    return cv.imdecode(np.frombuffer(buffer, np.uint8), cv.IMREAD_COLOR)


class BinaryMelanomaDetector:

    MODEL_PATH_NAME = "model_v0_train_val.pkl"


    def test(self, img):
        feat_dict, (img_post, mask_post) = self.extrac_features(img, return_mask=True)

        # color features selection
        feat_dict["image"]["color"] = feat_dict["image"]["color"][:,[0,1,2,3,5,9,10,11,12,14,15,16,19,22,26,28,29,31,32]]

        # classification
        classifier = Classifier()
        classifier.load(f"{get_base_folder()}models/{self.MODEL_PATH_NAME}")
        combined_pred, individual_pred, combined_prob, individual_prob  = classifier.test(feat_dict, return_proba=True)

        # draw mask on image
        img_result = fc.draw_mask(img_post, mask_post, thickness=5, mask_color=(0,255,0))

        # results
        if combined_pred[0] == 1:
            prob = combined_prob[0]
        else:
            prob =  1 - combined_prob[0]
        return combined_pred[0], prob, img_result



    def extrac_features(self, img_original, return_mask=False):
        # ------------ step 1: preprocessing - image preprocessed V0
        img_prep_v0 = outer_circle_remover.remove_outer_circle(img_original, new_bg_color=None)
        img_prep_v0 = jpg_compresion(img_prep_v0) # legacy code
        circular_mask = outer_circle_remover.get_preprocessed_circular_mask(img_original)

        # ------------ step 2: mask combined
        # -------- step 2.1: remove hairs
        img_no_hairs = hair_remover.remove_hairs(img_prep_v0)
        img_no_hairs = jpg_compresion(img_no_hairs) # legacy code
        # -------- step 2.2: watershed mask
        mask_wathershed = mask_watershed_creator.get_mask(img_no_hairs)
        mask_wathershed_ref  = refiner.refine(mask_wathershed, circular_mask, medianb_size=7, resize_w=512, gaussian_factor=6)
        # -------- step 2.3: probabilistic mask
        labels_meanshift = label_meanshift_creator.get_labels(img_no_hairs, resize_w=256)
        # labels_meanshift = jpg_compresion(labels_meanshift) # legacy code
        labels_meanshift_em234= label_em_creator.get_labels(labels_meanshift, em_ks=[2,3,4], resize_w=256)
        mask_v1 = mask_from_em234_labels_creator.get_best_prob_mask_V1(img_no_hairs, labels_meanshift_em234, circular_mask)
        mask_v1  = refiner.refine(mask_v1, circular_mask, medianb_size=7, resize_w=512, gaussian_factor=6)
        mask_v2 = mask_from_em234_labels_creator.get_best_prob_mask_V2(img_no_hairs, labels_meanshift_em234, circular_mask)
        mask_v2  = refiner.refine(mask_v2, circular_mask, medianb_size=0, resize_w=512, gaussian_factor=6)
        # -------- step 2.4: probabilistic mask
        mask_combined = mask_combinator.combine_masks([mask_wathershed_ref, mask_v1, mask_v2], img_no_hairs)


        # ------------ step 3: postprocessing
        img_post, mask_post = postprocessing.postprocess1(img_original, img_prep_v0, mask_combined)
        mask_post_inverted = cv.bitwise_not(mask_post)
        mask_post_border = mask_border_creator.get_border_mask(mask_post)


        # ------------ step 4: Features extractor
        # -------- step 4.1: image features
        # ---- step 4.1.1: color features
        feat_img_color = f_color.f_extract_color(img_post)
        # ---- step 4.1.2: lbp features
        gray_image = cv.cvtColor(img_prep_v0, cv.COLOR_BGR2GRAY)
        img_lbp = lbp_image_creator.get_multiple_lbp_img(gray_image, [1,2,3], 8).astype(np.uint8)
        img_lbp = postprocessing.postprocess1(img_original, img_lbp)
        feat_img_lbp = f_texture_lbp.f_extract_texture_lbp(img_lbp, args_dict={"multiple_radius":[1,2,3], "num_point": 8})
        # ---- step 4.1.3: glcm features
        feat_img_glcm = f_texture_glcm.f_extract_texture_glcm(img_post, args_dict={"gray_labels":None})

        # -------- step 4.2: mask features
        # ---- step 4.2.1: color features
        feat_mask_color = f_color.f_extract_color_mask(img_post, {"mask":mask_post})
        # ---- step 4.2.2: lbp features
        feat_mask_lbp = f_texture_lbp.f_extract_texture_lbp_mask(img_lbp, args_dict={"mask":mask_post, "multiple_radius":[1,2,3], "num_point": 8})
        # ---- step 4.2.3: glcm features
        feat_mask_glcm = f_texture_glcm.f_extract_texture_glcm_mask(img_post, args_dict={"mask":mask_post, "gray_labels":None})
        # ---- step 4.2.3: shape features
        feat_mask_shape = f_shape_mask.f_extract_shape(img_post, {"mask":mask_post})

        # -------- step 4.3: mask inverted features
        # ---- step 4.3.1: color features
        feat_mask_inv_color = f_color.f_extract_color_mask(img_post, {"mask":mask_post_inverted})
        # ---- step 4.3.2: lbp features
        feat_mask_inv_lbp = f_texture_lbp.f_extract_texture_lbp_mask(img_lbp, args_dict={"mask":mask_post_inverted, "multiple_radius":[1,2,3], "num_point": 8})
        # ---- step 4.3.3: glcm features
        feat_mask_inv_glcm = f_texture_glcm.f_extract_texture_glcm_mask(img_post, args_dict={"mask":mask_post_inverted, "gray_labels":None})


        # -------- step 4.3: mask border features
        # ---- step 4.3.1: color features
        feat_mask_border_color = f_color.f_extract_color_mask(img_post, {"mask":mask_post_border})
        # ---- step 4.3.2: lbp features
        feat_mask_border_lbp = f_texture_lbp.f_extract_texture_lbp_mask(img_lbp, args_dict={"mask":mask_post_border, "multiple_radius":[1,2,3], "num_point": 8})
        # ---- step 4.3.3: glcm features
        feat_mask_border_glcm = f_texture_glcm.f_extract_texture_glcm_mask(img_post, args_dict={"mask":mask_post_border, "gray_labels":None})


        # -------- step 4.4: spatial features
        # ---- step 4.3.1: color features
        feat_spatial_color = f_color.f_extract_color_mask_spatial(img_post, {"mask":mask_post})
        # ---- step 4.3.2: lbp features
        feat_spatial_lbp = f_texture_lbp.f_extract_texture_lbp_mask_spatial(img_lbp, args_dict={"mask":mask_post, "multiple_radius":[1,2,3], "num_point": 8})
        # ---- step 4.3.3: glcm features
        feat_spatial_glcm = f_texture_glcm.f_extract_texture_glcm_mask_spatial(img_post, args_dict={"mask":mask_post, "gray_labels":128})


        feat_dict = {
            "image": {
                "color": feat_img_color,
                "texture_lbp": feat_img_lbp,
                "texture_glcm": feat_img_glcm,
            },
            "mask": {
                "color": feat_mask_color,
                "texture_lbp": feat_mask_lbp,
                "texture_glcm": feat_mask_glcm,
                "shape": feat_mask_shape,
            },
            "mask_inv": {
                "color": feat_mask_inv_color,
                "texture_lbp": feat_mask_inv_lbp,
                "texture_glcm": feat_mask_inv_glcm,
            },
            "mask_border": {
                "color": feat_mask_border_color,
                "texture_lbp": feat_mask_border_lbp,
                "texture_glcm": feat_mask_border_glcm,
            },
            "spatial": {
                "color": feat_spatial_color,
                "texture_lbp": feat_spatial_lbp,
                "texture_glcm": feat_spatial_glcm,
            },
        }

        # Convert all elements of feat_dict to numpy arrays
        for key1 in feat_dict:
            for key2 in feat_dict[key1]:
                feat_dict[key1][key2] = np.array(feat_dict[key1][key2]).reshape(1, -1)

        if return_mask:
            img_no_hairs = postprocessing.postprocess1(img_original, img_no_hairs)

            return feat_dict, (img_no_hairs, mask_post)
        else:                          
            return feat_dict
