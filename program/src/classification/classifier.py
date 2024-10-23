
import numpy as np
import joblib
import threading

from .features_engeneer import FeaturesEngeneer
from .weak_classifier import WeakClassifier

from .classifier_combination import combiner


class Classifier:

    PARAMS_C1 = {'C': 10, 'degree': 5, 'gamma': 'scale', 'kernel': 'rbf', "probability":True, "random_state":42}
    PARAMS_C2 = {'C': 10, 'kernel': 'poly', 'gamma': 'scale', 'degree':5, "probability":True, "random_state":42}
    PARAMS_C3 = {'C': 100, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly', "probability":True, "random_state":42}
    PARAMS_C4 = {'C': 100, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly', "probability":True, "random_state":42}
    PARAMS_C5 = {'C': 100, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly', "probability":True, "random_state":42}

    CLASSIFIER_WEIGHTS = np.array([0.8587987355110642,0.8379873551106428,0.8163856691253951,0.7987355110642782,0.821654373024236])


    def __init__(self):
        self.features_engeneer = FeaturesEngeneer()
        self.c1 = WeakClassifier(self.PARAMS_C1)
        self.c2 = WeakClassifier(self.PARAMS_C2)
        self.c3 = WeakClassifier(self.PARAMS_C3)
        self.c4 = WeakClassifier(self.PARAMS_C4)
        self.c5 = WeakClassifier(self.PARAMS_C5)

        self.trained = False
        

    # Multithreaded training
    def train(self, features_dict, labels):

        # feature engineering and get features for classifiers  
        features_dict_pca, features_dict_scaled_image = self.features_engeneer.fit_transform(features_dict)
        x_all_pca, x_image, x_image_pca, x_mask_pca, x_spatial_pca = self.__get_features_for_classifier(features_dict_pca, features_dict_scaled_image)

        # training functions
        def train_c1():
            print("fit C1")
            self.c1.fit(x_all_pca, labels)
            print("end C1")

        def train_c2():
            print("fit C2")
            self.c2.fit(x_image, labels)
            print("end C2")

        def train_c3():
            print("fit C3")
            self.c3.fit(x_image_pca, labels)
            print("end C3")

        def train_c4():
            print("fit C4")
            self.c4.fit(x_mask_pca, labels)
            print("end C4")

        def train_c5():
            print("fit C5")
            self.c5.fit(x_spatial_pca, labels)
            print("end C5")

        # create threads
        threads = [
            threading.Thread(target=train_c1),
            threading.Thread(target=train_c2),
            threading.Thread(target=train_c3),
            threading.Thread(target=train_c4),
            threading.Thread(target=train_c5)
        ]

        # start all threads
        for thread in threads:
            thread.start()

        # wait for all threads to finish
        for thread in threads:
            thread.join()

        self.trained = True


    def test(self, features_dict, return_proba=False):
        if not self.trained:
            raise Exception('Classificator not trained')
        
        # feature engineering and get features for classifiers
        features_dict_pca, features_dict_scaled_image = self.features_engeneer.transform(features_dict)
        x_all_pca, x_image, x_image_pca, x_mask_pca, x_spatial_pca = self.__get_features_for_classifier(features_dict_pca, features_dict_scaled_image)
        
        return self.__predict_combined(x_all_pca, x_image, x_image_pca, x_mask_pca, x_spatial_pca, return_proba=return_proba)


    def __predict_combined(self, x_all_pca, x_image, x_image_pca, x_mask_pca, x_spatial_pca, return_proba=False):
        
        scores = np.zeros((5, x_all_pca.shape[0]))
        c_predictions = np.zeros((5, x_all_pca.shape[0]))
        
        c_predictions[0], scores[0] = self.c1.predict(x_all_pca)
        c_predictions[1], scores[1] = self.c2.predict(x_image)
        c_predictions[2], scores[2] = self.c3.predict(x_image_pca)
        c_predictions[3], scores[3] = self.c4.predict(x_mask_pca)
        c_predictions[4], scores[4] = self.c5.predict(x_spatial_pca)

        combined_scores = combiner.combine_results(scores, self.CLASSIFIER_WEIGHTS, mass_method=combiner.MASS_LINEAR, 
                                            combination_method=combiner.COMBINATION_DEMPSTER, 
                                            weights_influence=0)
        # ---- compute results
        y_combined = np.where(combined_scores > 0.5, 1, 0)
        
        if return_proba:
            return y_combined, c_predictions, combined_scores, scores
        else:
            return y_combined, c_predictions
    


    def __get_features_for_classifier(self, features_dict_pca, features_dict_scaled_image):
        # features for each classifier
        x_to_concat = []
        for feature_type, f_dict in features_dict_pca.items():
            if feature_type == "spatial":
                continue
            for feature_name, array in f_dict.items():
                x_to_concat.append(array)
        x_all_pca = np.hstack(x_to_concat)

        x_image = np.hstack([features_dict_scaled_image["image"]["color"], features_dict_scaled_image["image"]["texture_lbp"], features_dict_scaled_image["image"]["texture_glcm"]])
        x_image_pca = np.hstack([features_dict_pca["image"]["color"], features_dict_pca["image"]["texture_lbp"], features_dict_pca["image"]["texture_glcm"]])
        x_mask_pca = np.hstack([features_dict_pca["mask"]["color"], features_dict_pca["mask"]["texture_lbp"], features_dict_pca["mask"]["texture_glcm"]])
        x_spatial_pca = np.hstack([features_dict_pca["spatial"]["color"], features_dict_pca["spatial"]["texture_lbp"], features_dict_pca["spatial"]["texture_glcm"]])
        
        return x_all_pca, x_image, x_image_pca, x_mask_pca, x_spatial_pca
        

    def save(self, filepath):
        model_data = {
            'features_engeneer': self.features_engeneer,
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
            'c4': self.c4,
            'c5': self.c5
        }
        joblib.dump(model_data, filepath)
        print(f'Model saved in {filepath}')


    def load(self, filepath):
        model_data = joblib.load(filepath)
        self.features_engeneer = model_data['features_engeneer']
        self.c1 = model_data['c1']
        self.c2 = model_data['c2']
        self.c3 = model_data['c3']
        self.c4 = model_data['c4']
        self.c5 = model_data['c5']
        self.trained = True
        print(f'Model loaded from {filepath}')
