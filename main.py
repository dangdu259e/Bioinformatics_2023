import model
from model import *
from data import data
import feature_selection

if __name__ == '__main__':
    # load data
    path_data = 'data'
    X_train, y_train, X_test, y_test = data.load_data(path_data)
    # create column name for file save
    filename_result = "result.csv"
    write_result([['feature_selection_method', 'n_feature_to_select', 'model_name', 'sn', 'sp', 'acc', 'mcc', 'auc']],
                 filename_result)
    # feature selection using RFE
    feature_selection.RFE_MLR(X_train, y_train, X_test, y_test, filename_result)
    feature_selection.RFE_DT(X_train, y_train, X_test, y_test, filename_result)

    # feature selection using ratio
    filename_result_lg = "result(LR).csv"
    write_result([['feature_selection_method', 'ratio', 'model_name', 'sn', 'sp', 'acc', 'mcc', 'auc']],
                 filename_result_lg)
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        sn, sp, acc, mcc, auc = model.logistic_regression(X_train, y_train, X_test, y_test, ratio)
        result = [['LR', ratio, 'LR', sn, sp, acc, mcc, auc]]
        write_result(result, filename_result_lg)
