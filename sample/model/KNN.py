from sklearn.neighbors import KNeighborsClassifier
from sample.model.measure import *
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter
def KNN(X_train, y_train, X_test, y_test, neighbors=5):
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    sn, sp, acc, mcc, auc = cal_measure(y_test, y_pred)
    return sn, sp, acc, mcc, auc