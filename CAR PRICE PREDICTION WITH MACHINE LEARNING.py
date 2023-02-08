#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


#import all libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[3]:


cars_data = pd.read_csv('CarPrice.csv')
cars_data.head()


# In[4]:


#shape of the data
cars_data.shape


# In[5]:


#info the dataframe
cars_data.info()


# In[6]:


#describe the data
cars_data.describe(percentiles = [0.10,0.25,0.50,0.75,0.90,0.99])


# In[8]:


#cleaning the dataset to findout the duplicates
cars_data.duplicated(subset = ['car_ID']).sum()


# In[9]:


cars_data = cars_data.drop(['car_ID'], axis =1)


# In[11]:


#to find null values
cars_data.isnull().sum()


# In[12]:


#symboling column- Its assigned insurance risk rating, 
#A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.
cars_data['symboling'].value_counts()


# In[13]:


sns.pairplot(y_vars = 'symboling', x_vars = 'price' ,data = cars_data)


# In[14]:


#Column CarName
cars_data['CarName'].value_counts()


# In[15]:


cars_data['car_company'] = cars_data['CarName'].apply(lambda x:x.split(' ')[0])


# In[16]:


#rechecking
cars_data.head()


# In[17]:


#deleting the original column
cars_data = cars_data.drop(['CarName'], axis =1)


# In[18]:


cars_data['car_company'].value_counts()


# In[20]:


#correcting the spelling mistakes
cars_data['car_company'].replace('toyouta', 'toyota',inplace=True)
cars_data['car_company'].replace('Nissan', 'nissan',inplace=True)
cars_data['car_company'].replace('maxda', 'mazda',inplace=True)
cars_data['car_company'].replace('vokswagen', 'volkswagen',inplace=True)
cars_data['car_company'].replace('vw', 'volkswagen',inplace=True)
cars_data['car_company'].replace('porcshce', 'porsche',inplace=True)


# In[21]:


#rechecking the data:
cars_data['car_company'].value_counts()


# In[22]:


# fueltype - Car fuel type i.e gas or diesel
cars_data['fueltype'].value_counts()


# In[23]:


#aspiration - Aspiration used in a car
cars_data['aspiration'].value_counts()


# In[24]:


#doornumber - Number of doors in a car
cars_data['doornumber'].value_counts()


# In[66]:


#number to be converted into numeric form
def number_(x):
    return x.map({'four':4, 'two': 2})
    
cars_data['doornumber'] = cars_data[['doornumber']].apply(number_)


# In[67]:


#rechecking
cars_data['doornumber'].value_counts()


# In[49]:


#carbody- body of car
cars_data['carbody'].value_counts()


# In[50]:


#drivewheel - type of drive wheel
cars_data['drivewheel'].value_counts()


# In[51]:


#enginelocation - Location of car engine
cars_data['enginelocation'].value_counts()


# In[52]:


#wheelbase - Weelbase of car 
cars_data['wheelbase'].value_counts().head()


# In[53]:


sns.distplot(cars_data['wheelbase'])
plt.show()


# In[54]:


#carlength - Length of car
cars_data['carlength'].value_counts().head()


# In[55]:


sns.distplot(cars_data['carlength'])
plt.show()


# In[56]:


#enginetype - Type of engine.
cars_data['enginetype'].value_counts()


# In[57]:


#cylindernumber- cylinder placed in the car
cars_data['cylindernumber'].value_counts()


# In[58]:


#numbers to be converted into numeric form
def convert_number(x):
    return x.map({'two':2, 'three':3, 'four':4,'five':5, 'six':6,'eight':8,'twelve':12})

cars_data['cylindernumber'] = cars_data[['cylindernumber']].apply(convert_number)


# In[59]:


#re-checking
cars_data['cylindernumber'].value_counts()


# In[60]:


#fuelsystem - Fuel system of car
cars_data['fuelsystem'].value_counts()


# In[61]:


#data visulaization to look for any patterns
cars_numeric = cars_data.select_dtypes(include =['int64','float64'])
cars_numeric.head()


# In[62]:


plt.figure(figsize = (20,20))
sns.pairplot(cars_numeric)
plt.show()


# In[63]:


plt.figure(figsize = (20,20))
sns.heatmap(cars_data.corr(), annot = True ,cmap = 'YlGnBu')
plt.show()


# In[ ]:


#Price is highly (positively) correlated with wheelbase, carlength, carwidth, curbweight, enginesize, horsepower.

#Price is negatively correlated to symboling, citympg and highwaympg.

#This suggest that cars having high mileage may fall in the 'economy' cars category, and are priced lower.

#There are many independent variables which are highly correlated: wheelbase, carlength, curbweight, enginesize etc.. all are positively correlated.


# In[64]:


categorical_cols = cars_data.select_dtypes(include = ['object'])
categorical_cols.head()


# In[65]:


plt.figure(figsize = (20,12))
plt.subplot(3,3,1)
sns.boxplot(x = 'fueltype', y = 'price', data = cars_data)
plt.subplot(3,3,2)
sns.boxplot(x = 'aspiration', y = 'price', data = cars_data)
plt.subplot(3,3,3)
sns.boxplot(x = 'carbody', y = 'price', data = cars_data)
plt.subplot(3,3,4)
sns.boxplot(x = 'drivewheel', y = 'price', data = cars_data)
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = cars_data)
plt.subplot(3,3,6)
sns.boxplot(x = 'enginetype', y = 'price', data = cars_data)
plt.subplot(3,3,7)
sns.boxplot(x = 'fuelsystem', y = 'price', data = cars_data)


# In[68]:


plt.figure(figsize = (20,12))
sns.boxplot(x = 'car_company', y = 'price', data = cars_data)


# In[69]:


#creating dummies
cars_dummies = pd.get_dummies(categorical_cols, drop_first = True)
cars_dummies.head()


# In[70]:


car_df  = pd.concat([cars_data, cars_dummies], axis =1)


# In[71]:


car_df = car_df.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',
       'enginetype', 'fuelsystem', 'car_company'], axis =1)


# In[72]:


car_df.info()


# In[73]:


df_train, df_test = train_test_split(car_df, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[74]:


df_train.shape


# In[75]:


df_test.shape


# In[76]:


col_list = ['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth','carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',
            'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']


# In[77]:


scaler = StandardScaler()


# In[78]:


df_train[col_list] = scaler.fit_transform(df_train[col_list])


# In[93]:


df_train.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




