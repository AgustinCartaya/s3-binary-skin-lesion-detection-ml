
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class FeaturesEngeneer:
    
    def __init__(self):
        self.min_max_scaler_dict = {
            "image": {
                "color": None,
                "texture_lbp": None,
                "texture_glcm": None,
            },
            "mask": {
                "color": None,
                "texture_lbp": None,
                "texture_glcm": None,
                "shape": None,
            },
            "mask_inv": {
                "color": None,
                "texture_lbp": None,
                "texture_glcm": None,
            },
            "mask_border": {
                "color": None,
                "texture_lbp": None,
                "texture_glcm": None,
            },
            "spatial": {
                "color": None,
                "texture_lbp": None,
                "texture_glcm": None,
            },
        }

        self.pca_dict =  copy.deepcopy(self.min_max_scaler_dict)
        self.trained = False



    def fit_transform(self, features_dict):
        features_dict_pca = copy.deepcopy(features_dict)
        features_dict_scaled_image = {"image":copy.deepcopy(features_dict["image"])}
        # fit min max scaler and pca
        for feature_type, feature_dict in features_dict.items():
            for feature_name, feature in feature_dict.items():
                _min_masx_sc = MinMaxScaler()
                _pca = PCA(n_components=feature.shape[1])

                _feat = _min_masx_sc.fit_transform(feature)
                if feature_type == "image":
                    features_dict_scaled_image["image"][feature_name] = _feat
                _feat = _pca.fit_transform(_feat)

                features_dict_pca[feature_type][feature_name] = _feat
                self.min_max_scaler_dict[feature_type][feature_name] = _min_masx_sc
                self.pca_dict[feature_type][feature_name] = _pca

        self.trained = True
        return features_dict_pca, features_dict_scaled_image



    def transform(self, features_dict):
        if not self.trained:
            raise Exception('FeaturesEngeneer not trained')

        features_dict_pca = copy.deepcopy(features_dict)
        features_dict_scaled_image = {"image":copy.deepcopy(features_dict["image"])}
        
        for feature_type, feature_dict in features_dict.items():
            for feature_name, feature in feature_dict.items():

                _feat = self.min_max_scaler_dict[feature_type][feature_name].transform(feature)

                if feature_type == "image":
                    features_dict_scaled_image["image"][feature_name] = _feat

                _feat = self.pca_dict[feature_type][feature_name].transform(_feat)

                features_dict_pca[feature_type][feature_name] = _feat

        return features_dict_pca, features_dict_scaled_image
