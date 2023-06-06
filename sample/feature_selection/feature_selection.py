from sample.data import *
from sample.model.DT import *
from sample.model.SVM import *
from sample.model.KNN import *
from sample.model.measure import *
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn import tree

# select feature using Linear Regression Model
RFE_regressor = LinearRegression()

# tạo tiêu đề cột file save
write_result([['feature_selection_method','n_feature_to_select', 'model_name', 'sn', 'sp', 'acc', 'mcc', 'auc']])

for i in range(1, len(X_train[0]) - 1):
    rfe = RFE(RFE_regressor, n_features_to_select=i)
    rfe.fit_transform(X_train, y_train)
    # print(rfe.support_)
    # print(rfe.ranking_)
    # Reduce X to the selected features.
    X_train_reduce = rfe.transform(X_train)
    X_test_reduce = rfe.transform(X_test)
    # DT
    sn, sp, acc, mcc, auc = DT(X_train_reduce, y_train, X_test_reduce, y_test)
    result = [['RFE_MLR', i, 'DT', sn, sp, acc, mcc, auc]]
    write_result(result)
    # SVM
    sn, sp, acc, mcc, auc = SVM(X_train_reduce, y_train, X_test_reduce, y_test)
    result = [['RFE_MLR', i, 'SVM', sn, sp, acc, mcc, auc]]
    write_result(result)
    # KNN
    sn, sp, acc, mcc, auc = KNN(X_train_reduce, y_train, X_test_reduce, y_test)
    result = [['RFE_MLR', i, 'KNN', sn, sp, acc, mcc, auc]]
    write_result(result)




# select feature using Linear Regression Model
RFE_DT = tree.DecisionTreeClassifier()
for i in range(1, len(X_train[0]) - 1):
    rfe = RFE(RFE_DT, n_features_to_select=i)
    rfe.fit_transform(X_train, y_train)
    # print(rfe.support_)
    # print(rfe.ranking_)
    # Reduce X to the selected features.
    X_train_reduce = rfe.transform(X_train)
    X_test_reduce = rfe.transform(X_test)
    # DT
    sn, sp, acc, mcc, auc = DT(X_train_reduce, y_train, X_test_reduce, y_test)
    result = [['RFE_DT', i, 'DT', sn, sp, acc, mcc, auc]]
    write_result(result)
    # SVM
    sn, sp, acc, mcc, auc = SVM(X_train_reduce, y_train, X_test_reduce, y_test)
    result = [['RFE_DT', i, 'SVM', sn, sp, acc, mcc, auc]]
    write_result(result)
    # KNN
    sn, sp, acc, mcc, auc = KNN(X_train_reduce, y_train, X_test_reduce, y_test)
    result = [['RFE_DT', i, 'KNN', sn, sp, acc, mcc, auc]]
    write_result(result)