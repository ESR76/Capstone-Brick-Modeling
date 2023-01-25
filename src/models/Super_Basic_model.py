#!/usr/bin/env python
# coding: utf-8

# In[50]:


import brickschema
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[51]:


#testing stuff


# In[52]:



df = pd.read_csv('merged_all_2F.csv')


# In[53]:


print(df['time'])


# In[110]:


def temp_conversion(val):
    return (val - 32) * 5/9 + 273.15

df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
df['Outside Air Temp'] = temp_conversion(df['Outside Air Temp'])
df['Zone Temperature'] = temp_conversion(df['Zone Temperature'])

#datestuff
df['year'] = df['time'].dt.year
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['minute'] = df['time'].dt.minute


# In[111]:


trainstart = df[:len(df)//2]
teststart = df[len(df)//2:]


# In[112]:


#cleaned data
train = trainstart[['Common Setpoint','Actual Sup Flow SP','Zone Temperature',                    'Actual Supply Flow','energy','Humidity', 'Outside Air Temp',                    'year','day','hour','month','minute']]
test = teststart[['Common Setpoint','Actual Sup Flow SP','Zone Temperature',                  'Actual Supply Flow','energy','Humidity', 'Outside Air Temp',                  'year','day','hour','month','minute']]


# In[113]:


train


# In[114]:


regr = LinearRegression()
#Xtrain = train.drop(columns=['energy'])
Xtrain = train.drop(columns=['energy'])
Ytrain = train[['energy']]
Xtest = test.drop(columns=['energy'])
Ytest = test[['energy']]


# In[115]:


regr.fit(Xtrain, Ytrain)


# In[116]:


y_pred = regr.predict(Xtest)


# In[117]:


y_pred


# In[118]:


Ytest


# In[119]:



mean_squared_error(np.array(Ytest), y_pred)


# In[132]:


regr = LinearRegression()
#Xtrain = train.drop(columns=['energy'])
Xtrain = train.drop(columns=['energy','month'])
Ytrain = train[['energy']]
Xtest = test.drop(columns=['energy','month'])
Ytest = test[['energy']]
regr.fit(Xtrain, Ytrain)
y_pred = regr.predict(Xtest)
mean_squared_error(np.array(Ytest), y_pred)
#month seems to worsen the result


# In[125]:





# In[ ]:




