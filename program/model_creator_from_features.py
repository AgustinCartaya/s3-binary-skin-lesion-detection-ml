
import pandas as pd
import pathlib
import os
import numpy as np

from sklearn.metrics import accuracy_score
from classification.classifier import Classifier

def get_base_folder():
    return str(pathlib.Path(__file__).parent.resolve()).replace(os.sep, "/") + "/"

    
class ModelCreatorFromFeatures:

    F_FOLDER_POSTPROCESSING_V0 = "postprocessing_v0"
    F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1 = "postprocessing_v0_mask_combined_v1"
    F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I = "postprocessing_v0_mask_combined_v1_i"
    F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER = "postprocessing_v0_mask_combined_v1_border"

    F_COLOR = "color"
    F_TEXTURE_LBP = "texture_lbp"
    F_TEXTURE_GLCM = "texture_glcm"
    F_SHAPE = "shape"

    F_COLOR_SPATIAL = "color_spatial"
    F_TEXTURE_GLCM_SPATIAL = "texture_glcm_spatial"
    F_TEXTURE_LBP_SPATIAL = "texture_lbp_spatial"

    MAX_INDEX_TRAIN_BENIGN = 7725
    MAX_INDEX_TRAIN_MALIGN = 7470

    MODEL_PATH_NAME = "model_v0_only_train.pkl"


    def train_from_features(self, use_val=False):
        # load train features and labels
        def load_train_features(folder, features_name):
            file_np = pd.read_csv(f"{get_base_folder()}features/train_val/{folder}/{features_name}.csv").values
            if use_val:
                features = file_np[:, 2:]
                return features
            else:
                data_ben = file_np[file_np[:, 1] == 0]
                data_mal = file_np[file_np[:, 1] == 1]
                features =  np.concatenate((data_ben[:self.MAX_INDEX_TRAIN_BENIGN,2:], data_mal[:self.MAX_INDEX_TRAIN_MALIGN,2:]), axis=0)
                # features =  np.concatenate((data_ben[:1000,2:], data_mal[:1000,2:]), axis=0)
                return features
        
        def load_train_labels():
            file_np = pd.read_csv(f"{get_base_folder()}features/train_val/{self.F_FOLDER_POSTPROCESSING_V0}/{self.F_COLOR}.csv").values
            if use_val:
                labels = file_np[:, 1]
                return labels
            else:
                data_ben = file_np[file_np[:, 1] == 0]
                data_mal = file_np[file_np[:, 1] == 1]
                labels =  np.concatenate((data_ben[:self.MAX_INDEX_TRAIN_BENIGN,1], data_mal[:self.MAX_INDEX_TRAIN_MALIGN,1]), axis=0)
                # labels =  np.concatenate((data_ben[:1000,1], data_mal[:1000,1]), axis=0)
                return labels


        labels = load_train_labels()
        train_features_dict = {
            "image": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_COLOR),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_TEXTURE_LBP),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_TEXTURE_GLCM),
            },
            "mask": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_COLOR),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_LBP),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_GLCM),
                "shape": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_SHAPE),
            },
            "mask_inv": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_COLOR),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_TEXTURE_LBP),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_TEXTURE_GLCM),
            },
            "mask_border": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_COLOR),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_TEXTURE_LBP),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_TEXTURE_GLCM),
            },
            "spatial": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_COLOR_SPATIAL),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_LBP_SPATIAL),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_GLCM_SPATIAL),
            },
        }


        # color features selection
        train_features_dict["image"]["color"] = train_features_dict["image"]["color"][:,[0,1,2,3,5,9,10,11,12,14,15,16,19,22,26,28,29,31,32]]

        # train
        classifier = Classifier()
        classifier.train(train_features_dict, labels)

        # save
        classifier.save(f"{get_base_folder()}models/{self.MODEL_PATH_NAME}")

        # test predictions
        labels_pred, individual_pred = classifier.test(train_features_dict)

        accuracy = accuracy_score(labels, labels_pred)
        print(f"Accuracy: {accuracy}")

        for i in range(len(individual_pred)):
            acc = accuracy_score(labels, individual_pred[i])
            print(f"Accuracy C{i+1}: {acc}")


    def val_from_features(self):
        # load val features and labels
        def load_val_features(folder, features_name):
            file_np = pd.read_csv(f"{get_base_folder()}features/train_val/{folder}/{features_name}.csv").values
            data_ben = file_np[file_np[:, 1] == 0]
            data_mal = file_np[file_np[:, 1] == 1]
            features = np.concatenate((data_ben[self.MAX_INDEX_TRAIN_BENIGN:,2:], data_mal[self.MAX_INDEX_TRAIN_MALIGN:,2:]), axis=0)
            return features
            
        def load_val_labels():
            file_np = pd.read_csv(f"{get_base_folder()}features/train_val/{self.F_FOLDER_POSTPROCESSING_V0}/{self.F_COLOR}.csv").values
            data_ben = file_np[file_np[:, 1] == 0]
            data_mal = file_np[file_np[:, 1] == 1]
            labels = np.concatenate((data_ben[self.MAX_INDEX_TRAIN_BENIGN:,1], data_mal[self.MAX_INDEX_TRAIN_MALIGN:,1]), axis=0)
            return labels

        labels = load_val_labels()
        val_features_dict = {
            "image": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_COLOR),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_TEXTURE_LBP),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_TEXTURE_GLCM),
            },
            "mask": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_COLOR),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_LBP),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_GLCM),
                "shape": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_SHAPE),
            },
            "mask_inv": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_COLOR),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_TEXTURE_LBP),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_TEXTURE_GLCM),
            },
            "mask_border": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_COLOR),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_TEXTURE_LBP),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_TEXTURE_GLCM),
            },
            "spatial": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_COLOR_SPATIAL),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_LBP_SPATIAL),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_GLCM_SPATIAL),
            },
        }

        # color features selection
        val_features_dict["image"]["color"] = val_features_dict["image"]["color"][:,[0,1,2,3,5,9,10,11,12,14,15,16,19,22,26,28,29,31,32]]

        # classification
        classifier = Classifier()
        classifier.load(f"{get_base_folder()}models/{self.MODEL_PATH_NAME}")
        labels_pred, individual_pred = classifier.test(val_features_dict)

        # results
        accuracy = accuracy_score(labels, labels_pred)
        print(f"Accuracy: {accuracy}")

        for i in range(len(individual_pred)):
            acc = accuracy_score(labels, individual_pred[i])
            print(f"Accuracy C{i+1}: {acc}")



if __name__ == '__main__':
    bin_mel_classification = ModelCreatorFromFeatures()
    # print("Train -------------------------------------------------")
    # bin_mel_classification.train_from_features(use_val=True)
    print("Val -------------------------------------------------")
    bin_mel_classification.val_from_features()


# validation accuracy using training features

# Accuracy: 0.875131717597471
# Accuracy C1: 0.8587987355110642
# Accuracy C2: 0.8379873551106428
# Accuracy C3: 0.8163856691253951
# Accuracy C4: 0.7987355110642782
# Accuracy C5: 0.821654373024236