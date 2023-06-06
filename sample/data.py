import pandas as pd

# Load dataset
X_train = pd.read_csv('../../data/X_train.csv').to_numpy()[1:,1:]
y_train = pd.read_csv('../../data/y_train.csv').to_numpy()[1:,1:].astype('int').ravel()
X_test = pd.read_csv('../../data/X_test.csv').to_numpy()[1:,1:]
y_test = pd.read_csv('../../data/y_test.csv').to_numpy()[1:,1:].astype('int').ravel()