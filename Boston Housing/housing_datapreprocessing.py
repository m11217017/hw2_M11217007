#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


Path = 'E:\\NYUST_CSIE_112學年度第1學期_Course\\資料探勘_許中川教授\\專案作業二\\Boston_Housing_datapreprocessing\\housing.csv'
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_data_df = pd.read_csv(Path, header = None, delimiter = r"\s+", names = column_names)
housing_data = [housing_data_df]

housing_data_df.head(10)


# In[4]:


housing_data_df.info()


# In[5]:


housing_data_df.shape


# In[6]:


housing_data_df.isnull().sum()


# In[7]:


housing_data_df.describe()


# In[8]:


correlation = housing_data_df.corr()
correlation.shape


# In[9]:


plt.figure(figsize = (14, 14))
sns.heatmap(correlation, cbar = False, square = True, fmt = '.2%', annot = True, cmap = 'Greens')


# In[10]:


sns.heatmap(housing_data_df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[11]:


sns.set_style('whitegrid')
sns.countplot(x = 'RAD', data = housing_data_df)


# In[12]:


sns.set_style('whitegrid')
sns.countplot(x = 'CHAS', data = housing_data_df)


# In[13]:


sns.set_style('whitegrid')
sns.countplot(x = 'CHAS', hue = 'RAD', data = housing_data_df, palette = 'RdBu_r')


# In[14]:


sns.distplot(housing_data_df['AGE'].dropna(), kde = False, color = 'darkred', bins = 30)


# In[15]:


sns.distplot(housing_data_df['CRIM'].dropna(), kde = False, color = 'darkorange', bins = 30)


# In[16]:


sns.distplot(housing_data_df['RM'].dropna(), kde = False, color = 'darkblue', bins = 30)


# In[17]:


X = housing_data_df.iloc[:, 0:13]
y = housing_data_df.iloc[:, -1]


# In[18]:


y = np.round(housing_data_df['MEDV'])

bestFeatures = SelectKBest(score_func = chi2, k = 10)
fit = bestFeatures.fit(X, y)
housing_data_dfscores = pd.DataFrame(fit.scores_)
housing_data_dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([housing_data_dfcolumns, housing_data_dfscores], axis = 1)
featureScores.columns = ['housing_data_Specs', 'housing_data_Score']
featureScores


# In[19]:


print(featureScores.nlargest(10, 'housing_data_Score'))


# In[20]:


housing_data_df.head(10)


# In[21]:


Min_Max_Scaler = preprocessing.MinMaxScaler()
column_sels = ['TAX', 'ZN', 'CRIM', 'B', 'AGE', 'RAD', 'LSTAT', 'INDUS']
X = housing_data_df.loc[:,column_sels]
y = housing_data_df['MEDV']
x = pd.DataFrame(data = Min_Max_Scaler.fit_transform(X), columns = column_sels)
x


# In[ ]:





# In[ ]:




