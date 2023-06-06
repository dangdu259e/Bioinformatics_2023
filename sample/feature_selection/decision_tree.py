from sample.data import *
from sample.model.DT import *
from sample.model.SVM import *
from sample.model.KNN import *
from sample.model.measure import *
from sklearn.feature_selection import RFE
from sklearn import tree
import seaborn as sns

# select feature using Linear Regression Model
RFE_DT = tree.DecisionTreeClassifier()

for i in range(1, len(X_train[0])-1):
    rfe = RFE(RFE_DT, n_features_to_select=i)
    rfe.fit_transform(X_train,y_train)
    # print(rfe.support_)
    # print(rfe.ranking_)
    # Reduce X to the selected features.
    X_train_reduce = rfe.transform(X_train)
    X_test_reduce = rfe.transform(X_test)
    # DT
    sn, sp, acc, mcc, auc = DT(X_train_reduce, y_train, X_test_reduce, y_test)
    result = [[i, 'DT', sn, sp, acc, mcc, auc]]
    write_result(result)
    #SVM
    sn, sp, acc, mcc, auc = SVM(X_train_reduce, y_train, X_test_reduce, y_test)
    result = [[i, 'SVM', sn, sp, acc, mcc, auc]]
    write_result(result)
    #KNN
    sn, sp, acc, mcc, auc = KNN(X_train_reduce, y_train, X_test_reduce, y_test)
    result = [[i, 'KNN', sn, sp, acc, mcc, auc]]
    write_result(result)


