import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np


def cal_measure(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Sensitivity (sn)
    sn = tp / (tp + fn)
    # Specificity (sp)
    sp = tn / (tn + fp)
    # Accuracy (acc)
    acc = (tp + tn) / (tp + fp + tn + fn)
    # Matthew correlation coefficient (MCC)
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fn) * (tn + fp) * (tp + fp) * (tn * fn))
    auc = roc_auc_score(y_true, y_pred)
    return sn, sp, acc, mcc, auc


def write_result(result):
    print("Saving: ", result)
    # create dataframe
    df = pd.DataFrame(result,
                      columns=['feature_selection_method', 'n_feature_to_select', 'model_name', 'sn', 'sp', 'acc',
                               'mcc', 'auc'])
    df.to_csv('result.csv', mode='a', index=False, header=False)

# write_result([['feature_selection_method','n_feature_to_select', 'model_name', 'sn', 'sp', 'acc', 'mcc', 'auc']])
