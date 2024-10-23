
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, log_loss

def compute_cumulative_cm(y_gt, y_pred, prev_cm=None):
    cm = confusion_matrix(y_gt, y_pred)
    if prev_cm is None:
        return cm
    return cm + prev_cm

def compute_scores_from_cm(cm):
    # calculate the cofusion matrix
    tn, fp, fn, tp = cm.ravel()
    # Accuracy
    acc = (tp+tn)/(tn + fp + fn + tp)
    # specificity
    spec = tn / (tn + fp)
    # sensitivity (recall)
    sens = tp / (tp + fn)
    # precition
    if tp+fp > 0:
        prec = tp/(tp+fp)  
    else:
        prec = 0
        print("prec = tp/(tp+fp) ERROR ... TP + FP = 0")
    # f1
    if tp > 0:
        f1 = (2 * prec * sens)/(prec + sens)
    else:
        f1 = 0
    # ba
    ba = (spec + sens)/2

    return {"tn":tn, "fp":fp, "fn":fn, "tp":tp, "acc":acc, "spec":spec, "sens":sens, "prec":prec, "f1":f1, "ba":ba}


def compute_scores(y, y_pred):
    return compute_scores_from_cm(confusion_matrix(y, y_pred))


def compute_cross_entropy(y, y_pred):
    return log_loss(y, y_pred)