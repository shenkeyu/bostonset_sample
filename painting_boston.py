"""
作者：sky
版本：1.0
作用：使用波士顿数据进行数据分析和画图
时间：20181001
使用要求：放在anaconda中使用，不然sklearn没有datasets，则没有波士顿数据
"""
import matplotlib
import sklearn

boston_dataset = sklearn.datasets.load_boston()

# 打印波士顿dataset的描述 print(boston_dataset.DESCR)

#对数据进行设置
X_full = boston_dataset.data
Y = boston_dataset.target
print(X_full.shape)
print(Y.shape)

#使用SelectKBest类作为特征运算
selector = sklearn.SelectKBest(f_regression, k=1)
#使用线性模型
selector.fit(X_full, Y)
X = X_full[:, selector.get_support()]
print(X.shape)
#图形描绘
matplotlib.pyplot.scatter(X, Y, color='black')
matplotlib.pyplot.show()

#使用线性回归模型
regressor = sklearn.LinearRegression(normalize=True)
regressor.fit(X, Y)
#图形描绘
matplotlib.pyplot.scatter(X, Y, color='black')
matplotlib.pyplot.plot(X, regressor.predict(X), color='blue', linewidth=3)
matplotlib.pyplot.show()

#使用向量机模型
regressor = sklearn.SVR()
regressor.fit(X, Y)
#图形描绘
matplotlib.pyplot.scatter(X, Y, color='black')
matplotlib.pyplot.plot(X, regressor.predict(X), color='blue', linewidth=3)
matplotlib.pyplot.show()

#使用随机森林模型
regressor = sklearn.RandomForestRegressor()
regressor.fit(X, Y)
#图形描绘
matplotlib.pyplot.scatter(X, Y, color='black')
matplotlib.pyplot.plot(X, regressor.predict(X), color='blue', linewidth=3)
matplotlib.pyplot.show()