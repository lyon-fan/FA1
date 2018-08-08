import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
# 读取数据
data = pd.read_csv('data/santander.csv', nrows=50000)
data.shape
data.head()

# 我们只用数值变量进行演示

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]
data.shape

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['TARGET', 'ID'], axis=1), data['TARGET'], test_size=0.3, random_state=0)
X_train.shape, X_test.shape

# 训练随机森林
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
#  inter-trees variability. 衡量所有树中该特征的波动性
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
print("Feature ranking:")

feat_labels = X_train.columns
for f in range(X_train.shape[1]):
    print("%d. feature no:%d feature name:%s (%f)" % (f + 1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
# 对 top 15的特征作图。
# 纵轴是重要性，黑色竖线是在所有tree上的波动(标准差)
indices_top15 = indices[0:15]
plt.figure()
plt.title("Feature importances")
plt.bar(range(15), importances[indices_top15],
        color="r", yerr=std[indices_top15], align="center")
plt.xticks(range(15), indices_top15)
plt.xlim([-1,15])
plt.show()

