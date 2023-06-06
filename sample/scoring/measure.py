import math

from sklearn.metrics import confusion_matrix

# input y_true, y_predict
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

    # True negative rate (TNR)
    tnr = sp
    # False positive rate (FPR)
    fpr = 1 - sp
    for i in range(len(fpr) - 1)):
        AUC = sum((tpr[i+1] + tpr[i]) * (fpr[i+1] - fpr[i]) / 2)

    return sn, sp, acc, mcc

# test driver code
# y_true = [0, 0, 0, 1, 1, 1, 1, 1]
# y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
#
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# # Sensitivity (sn)
# sn = tp / (tp + fn)
# # Specificity (sp)
# sp = tn / (tn + fp)
# # Accuracy (acc)
# acc = (tp + tn) / (tp + fp + tn + fn)
# # Matthew correlation coefficient (MCC)
# mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fn) * (tn + fp) * (tp + fp) * (tn * fn))
#
# print(sn)
# print(sp)
# print(acc)
# print(mcc)