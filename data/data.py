import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# Load dataset
def raw_data(path):
    X_train = pd.read_csv(path + '/X_train.csv').to_numpy()[1:, 1:]
    y_train = pd.read_csv(path + '/y_train.csv').to_numpy()[1:, 1:].astype('int').ravel()
    X_test = pd.read_csv(path + '/X_test.csv').to_numpy()[1:, 1:]
    y_test = pd.read_csv(path + '/y_test.csv').to_numpy()[1:, 1:].astype('int').ravel()
    return X_train, y_train, X_test, y_test


# PCA
def get_pca(X_train, X_test, k=5):
    # PCA for all genotype
    # remove feature 'sex' in dataset
    X_train = X_train[:, :-1]
    X_test = X_test[:, :-1]

    pca = PCA(n_components=k, random_state=0)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    train_pca = pd.DataFrame(X_train_pca, columns=["PC" + str(i) for i in range(k)])
    test_pca = pd.DataFrame(X_test_pca, columns=["PC" + str(i) for i in range(k)])
    return train_pca, test_pca


def load_data(path):
    X_train, y_train, X_test, y_test = raw_data(path)
    train_pca, test_pca = get_pca(X_train, X_test, k=5)
    train_pca = train_pca.to_numpy()
    test_pca = test_pca.to_numpy()
    # create X_train_new = X_train + 5 component from PCA
    X_train_new = np.append(X_train, train_pca, axis=1)
    # create X_test_new
    X_test_new = np.append(X_test, test_pca, axis=1)
    return X_train_new, y_train, X_test_new, y_test
