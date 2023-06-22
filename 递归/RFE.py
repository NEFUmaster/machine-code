from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# 导入数据，路径中要么用\\或/或者在路径前加r
dataset = pd.read_csv(r'E:\jupyter\大兴安岭数据处理\train.csv')
# 输出数据预览
print(dataset)
# 准备训练数据
# 自变量：
# 因变量：
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
#estimator = RandomForestRegressor(n_estimators=200,max_depth=3,min_samples_leaf=2,min_samples_split=2,random_state=1)
estimator = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.19, max_delta_step=0,
             max_depth=1, min_child_weight=1,
             monotone_constraints='()', n_estimators=50, n_jobs=-1,
             num_parallel_tree=2, predictor='auto', random_state=1, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
#estimator = SVR(kernel='rbf',C=100,gamma=0.001).fit(X_train,y_train)
selector = RFE(estimator, n_features_to_select=1)
selector = selector.fit(X, y)

# 哪些特征入选最后特征，true表示入选
print(selector.support_)
# 每个特征的得分排名，特征得分越低（1最好），表示特征越好
print(selector.ranking_)
#  挑选了几个特征
print(selector.n_features_)



