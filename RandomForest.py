# _*_coding : UTF-8 _*_
# @Time : 2023/6/13 14:51 
# @Author : NEFU_GaoMY
# @File : RandomForest 
# @Project : machine learning
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.metrics import confusion_matrix
data = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\train.csv')
x = data.iloc[:, 0:97]
y = data.iloc[:, -1]

x_train = pd.DataFrame(x)
y_train = y.values.ravel()

rf_classifier = RandomForestClassifier(n_estimators=150, random_state=20)
rf_classifier.fit(x_train, y_train)

test_data = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\test.csv')
x_test = test_data.iloc[:, 0:97]
y_test = test_data.iloc[:, -1]

y_pred = rf_classifier.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
OA = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("kappa:", kappa, "OA:", OA)
print(classification_report(y_test, y_pred))

