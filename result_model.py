from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from data import data
import pandas as pd

# load data
path_data = 'data'
X_train, y_train, X_test, y_test = data.load_data(path_data)
labels = ['rs11208659', 'rs3101337', 'rs3101336', 'rs9425089', 'rs2568958', 'rs2815752', 'rs10789336', 'rs4322186',
          'rs1514176',
          'rs1514175', 'rs12410097', 'rs1555543', 'rs12021920', 'rs984222', 'rs1011731', 'rs543874', 'rs10913469',
          'rs2605100', 'rs939582',
          'rs2867124', 'rs6548238', 'rs6745262', 'rs4854343', 'rs4854344', 'rs4854345', 'rs7561317', 'rs713586',
          'rs887913', 'rs17049906',
          'rs2943650', 'rs4684846', 'rs1822825', 'rs6795735', 'rs7647305', 'rs9816226', 'rs10938397', 'rs7687015',
          'rs1800592', 'rs2112347',
          'rs6861681', 'rs1294421', 'rs206936', 'rs6905288', 'rs987237', 'rs2800710', 'rs9491696', 'rs1055144',
          'rs17150703', 'rs13278851',
          'rs13252210', 'rs516175', 'rs545854', 'rs4994', 'rs4735692', 'rs58104805', 'rs10968576', 'rs10508504',
          'rs4929949', 'rs4074134',
          'rs4923461', 'rs925946', 'rs10501087', 'rs6265', 'rs10767664', 'rs3817334', 'rs7120548', 'rs10838738',
          'rs564343', 'rs660339',
          'rs5443', 'rs718314', 'rs1948149', 'rs7138803', 'rs1443512', 'rs4759309', 'rs11109072', 'rs7316835',
          'rs2074356', 'rs17089410',
          'rs7989336', 'rs1957893', 'rs79090609', 'rs1957894', 'rs2241423', 'rs2531995', 'rs10163244', 'rs11860225',
          'rs4786083',
          'rs10500331', 'rs8052357', 'rs11643187', 'rs11646906', 'rs12924838', 'rs1946127', 'rs11077019', 'rs8049439',
          'rs4788102',
          'rs7498665', 'rs7359397', 'rs6499640', 'rs9939973', 'rs9940128', 'rs1421085', 'rs1558902', 'rs1121980',
          'rs72803680', 'rs7193144',
          'rs8050136', 'rs8051591', 'rs9935401', 'rs3751812', 'rs9926289', 'rs9939609', 'rs7190492', 'rs9930501',
          'rs9930506', 'rs9932754',
          'rs8044769', 'rs1424233', 'rs9890502', 'rs7503807', 'rs1805081', 'rs571480', 'rs571312', 'rs17782313',
          'rs476828', 'rs12970134',
          'rs477181', 'rs502933', 'rs4450508', 'rs29941', 'rs442398', 'rs11084753', 'rs13041126', 'rs4823006', 'Sex',
          'Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5']

RFE_regressor = LinearRegression()
RFE_DT = tree.DecisionTreeClassifier()


def feature_selected(estimator, n_features_to_select, name):
    rfe = RFE(estimator, n_features_to_select=n_features_to_select)
    rfe.fit_transform(X_train, y_train)
    mass = rfe.support_
    feature_selected = [labels[i] for i in range(len(labels)) if mass[i] == True]
    # print(feature_selected)
    # create dataframe
    save_feature = [[name, n_features_to_select, feature_selected]]
    df = pd.DataFrame(save_feature, columns=['name', 'n_features_to_select', 'feature_selected'])
    df.to_csv("feature_selected.csv", mode='a', index=False, header=False)


df = pd.DataFrame([['name', 'n_features_to_select', 'feature_selected']],columns=['name', 'n_features_to_select', 'feature_selected'])
df.to_csv("feature_selected.csv", mode='a', index=False, header=False)

feature_selected(RFE_DT, 45, 'RFE_DT')
feature_selected(RFE_DT, 3, 'RFE_DT')
feature_selected(RFE_DT, 4, 'RFE_DT')
feature_selected(RFE_DT, 2, 'RFE_DT')
feature_selected(RFE_regressor, 56, 'RFE_regressor')
feature_selected(RFE_regressor, 27, 'RFE_regressor')
feature_selected(RFE_regressor, 8, 'RFE_regressor')
