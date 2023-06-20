# _*_coding : UTF-8 _*_
# @Time : 2023/6/20 21:53 
# @Author : NEFU_GaoMY
# @File : 支持向量机 
# @Project : 递归特征消除
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 读取训练集数据
train_data = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\train.csv')
x_train = train_data.iloc[:, 0:35]  # 特征数据
y_train = train_data.iloc[:, -1]  # 目标数据

# 创建SVM分类器
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')

# 训练模型
svm_classifier.fit(x_train, y_train)

# 读取验证集数据
test_data = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\test.csv')
x_test = test_data.iloc[:, 0:35]
y_test = test_data.iloc[:, -1]

# 使用训练好的模型进行预测
y_pred = svm_classifier.predict(x_test)

# 计算模型的性能指标
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

# 输出结果
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
