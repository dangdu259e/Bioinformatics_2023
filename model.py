from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import math
import pandas as pd

def DT(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    sn, sp, acc, mcc, auc = cal_measure(y_test, y_pred)
    return sn, sp, acc, mcc, auc


import warnings

warnings.filterwarnings('ignore')  # setting ignore as a parameter


def KNN(X_train, y_train, X_test, y_test, neighbors=5):
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    sn, sp, acc, mcc, auc = cal_measure(y_test, y_pred)
    return sn, sp, acc, mcc, auc


def SVM(X_train, y_train, X_test, y_test):
    clf = svm.SVC()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    sn, sp, acc, mcc, auc = cal_measure(y_test, y_pred)
    return sn, sp, acc, mcc, auc

def logistic_regression(X_train, y_train, X_test, y_test, ratio):
    clf = LogisticRegression(random_state=0, penalty='elasticnet', l1_ratio=ratio, solver="saga", n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    sn, sp, acc, mcc, auc = cal_measure(y_test, y_pred)
    return sn, sp, acc, mcc, auc

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


def write_result(result, filename):
    print("Saving: ", result)
    # create dataframe
    df = pd.DataFrame(result,
                      columns=['feature_selection_method', 'n_feature_to_select', 'model_name', 'sn', 'sp', 'acc',
                               'mcc', 'auc'])
    df.to_csv(filename, mode='a', index=False, header=False)

# write_result([['feature_selection_method','n_feature_to_select', 'model_name', 'sn', 'sp', 'acc', 'mcc', 'auc']])
