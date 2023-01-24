#!/usr/bin/env python
# coding: utf-8

# In[106]:


import brickschema
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[107]:


#testing stuff


# In[108]:



df = pandas.read_csv('merged_all_2F.csv')


# In[109]:


print(df['time'])


# In[114]:


def temp_conversion(val):
    return (val - 32) * 5/9 + 273.15

df['time'] = df['time'].transform(pd.Timestamp)
df['Outside Air Temp'] = temp_conversion(df['Outside Air Temp'])
saf = df['Actual Supply Flow']
df['Zone Temperature'] = temp_conversion(df['Zone Temperature'])


# In[115]:


trainstart = df[:len(df)//2]
teststart = df[len(df)//2:]


# In[116]:


#cleaned data
train = trainstart[['time','Common Setpoint','Actual Sup Flow SP','Zone Temperature','Actual Supply Flow','energy','Humidity', 'Outside Air Temp']]
test = teststart[['time','Common Setpoint','Actual Sup Flow SP','Zone Temperature','Actual Supply Flow','energy','Humidity', 'Outside Air Temp']]


# In[117]:


train


# In[ ]:





# In[120]:


regr = LinearRegression()
#Xtrain = train.drop(columns=['energy'])
Xtrain = train.drop(columns=['energy','time'])
Ytrain = train[['energy']]
Xtest = test.drop(columns=['energy','time'])
Ytest = test[['energy']]


# In[121]:


regr.fit(Xtrain, Ytrain)


# In[122]:


y_pred = regr.predict(Xtest)


# In[123]:


y_pred


# In[124]:


Ytest


# In[125]:



mean_squared_error(np.array(Ytest), y_pred)


# In[ ]:





# In[ ]:




