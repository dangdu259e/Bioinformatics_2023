from model import *
from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.linear_model import LinearRegression




# Select feature using Linear Regression Model
def RFE_MLR(X_train, y_train, X_test, y_test, filename_result):
    RFE_regressor = LinearRegression()
    for i in range(1, len(X_train[0]) - 1):
        rfe = RFE(RFE_regressor, n_features_to_select=i)
        rfe.fit_transform(X_train, y_train)
        # Reduce X to the selected features.
        X_train_reduce = rfe.transform(X_train)
        X_test_reduce = rfe.transform(X_test)
        # DT
        sn, sp, acc, mcc, auc = DT(X_train_reduce, y_train, X_test_reduce, y_test)
        result = [['RFE_MLR', i, 'DT', sn, sp, acc, mcc, auc]]
        write_result(result, filename_result)
        # SVM
        sn, sp, acc, mcc, auc = SVM(X_train_reduce, y_train, X_test_reduce, y_test)
        result = [['RFE_MLR', i, 'SVM', sn, sp, acc, mcc, auc]]
        write_result(result, filename_result)
        # KNN
        sn, sp, acc, mcc, auc = KNN(X_train_reduce, y_train, X_test_reduce, y_test)
        result = [['RFE_MLR', i, 'KNN', sn, sp, acc, mcc, auc]]
        write_result(result, filename_result)


# Select features using decision tree model
def RFE_DT(X_train, y_train, X_test, y_test, filename_result):
    RFE_DT = tree.DecisionTreeClassifier()
    for i in range(1, len(X_train[0]) - 1):
        rfe = RFE(RFE_DT, n_features_to_select=i)
        rfe.fit_transform(X_train, y_train)
        # Reduce X to the selected features.
        X_train_reduce = rfe.transform(X_train)
        X_test_reduce = rfe.transform(X_test)

        # Predicted model
        # Decision tree model
        sn, sp, acc, mcc, auc = DT(X_train_reduce, y_train, X_test_reduce, y_test)
        result = [['RFE_DT',i, 'DT', sn, sp, acc, mcc, auc]]
        write_result(result, filename_result)
        # Support Vector Machine model
        sn, sp, acc, mcc, auc = SVM(X_train_reduce, y_train, X_test_reduce, y_test)
        result = [['RFE_DT',i, 'SVM', sn, sp, acc, mcc, auc]]
        write_result(result, filename_result)
        # K-Nearest Neighbors model
        sn, sp, acc, mcc, auc = KNN(X_train_reduce, y_train, X_test_reduce, y_test)
        result = [['RFE_DT',i, 'KNN', sn, sp, acc, mcc, auc]]
        write_result(result, filename_result)