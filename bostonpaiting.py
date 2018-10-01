
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
boston_dataset=datasets.load_boston()
X_full=boston_dataset.data
Y=boston_dataset.target
print(X_full.shape)
print(Y.shape)


# In[8]:


selector = SelectKBest(f_regression, k=1)
selector.fit(X_full, Y)
X = X_full[:, selector.get_support()]
print(X.shape)


# In[9]:


plt.scatter(X, Y, color='black')
plt.show()


# In[11]:


#使用回归模型
regressor = LinearRegression(normalize=True)
regressor.fit(X, Y)
#图形描绘
plt.scatter(X, Y, color='black')
plt.plot(X, regressor.predict(X), color='blue', linewidth=3)
plt.show()


# In[14]:


#使用向量机模型
regressor = SVR()
regressor.fit(X, Y)
#图形描绘
plt.scatter(X, Y, color='black')
plt.plot(X, regressor.predict(X), color='blue', linewidth=3)
plt.show()


# In[16]:


#使用随机森林模型
regressor = RandomForestRegressor()
regressor.fit(X, Y)
#图形描绘
plt.scatter(X, Y, color='black')
plt.plot(X, regressor.predict(X), color='blue', linewidth=3)
plt.show()

