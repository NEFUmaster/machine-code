# _*_coding : UTF-8 _*_
# @Time : 2023/6/13 15:01 
# @Author : NEFU_GaoMY
# @File : lightGBM 
# @Project : machine learning
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
# 读取训练数据
data = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\train.csv')
x = data.iloc[:, 0:97]
y = data.iloc[:, -1]

x_train = pd.DataFrame(x)
y_train = pd.Series(y.values.ravel())  # 将y转换为一维数组并改为Series类型

# 创建并训练LightGBM分类器
lgb_classifier = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=150)
lgb_classifier.fit(x_train, y_train)

# 读取测试数据
test_data = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\test.csv')
x_test = test_data.iloc[:, 0:97]
y_test = test_data.iloc[:, -1]

# 使用训练好的模型进行预测
y_pred = lgb_classifier.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
# 计算准确率和打印分类报告
print("Confusion Matrix:")
print(conf_matrix)
OA = accuracy_score(y_test, y_pred)
print("OA:", OA)
print(classification_report(y_test, y_pred))


