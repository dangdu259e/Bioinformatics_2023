import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors, datasets
from sklearn import metrics

X_train = pd.read_csv('./data/X_train.csv').to_numpy()[1:,1:]
y_train = pd.read_csv('./data/y_train.csv').to_numpy()[1:,1:].astype('int').ravel()
X_test = pd.read_csv('./data/X_test.csv').to_numpy()[1:,1:]
y_test = pd.read_csv('./data/y_test.csv').to_numpy()[1:,1:].astype('int').ravel()

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
print(metrics.roc_curve(y_test, y_pred, pos_label=2))
print(metrics.auc(fpr, tpr))
