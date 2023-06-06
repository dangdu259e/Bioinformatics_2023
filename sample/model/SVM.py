from sklearn import svm
from sample.model.measure import *
def SVM(X_train, y_train, X_test, y_test):
    clf = svm.SVC()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    sn, sp, acc, mcc, auc = cal_measure(y_test, y_pred)
    return sn, sp, acc, mcc, auc