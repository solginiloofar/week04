#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as pet
import matplotlib.pyplot as plt
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
data_train=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_train_data.csv')
dat_valid=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_valid_data.csv')
data_test=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_test_data.csv')
data_all=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\kc_house_data.csv')
data_set1=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_set_1_data.csv')
data_set2=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_set_2_data.csv')
data_set3=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_set_3_data.csv')
data_set4=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_set_4_data.csv')


# In[8]:


x=data_train['sqft_living']
y=data_train['price']
from scipy.interpolate import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels as s
from numpy.polynomial import polynomial
import matplotlib.pyplot as plt
import scipy


# In[9]:


get_ipython().magic(u'matplotlib inline')


# In[10]:


p1=np.polyfit(x,y,1)
print(p1)


# In[5]:


p2=np.polyfit(x,y,2)
p3=np.polyfit(x,y,3)
plt.plot(x,y,'o')
plt.plot(x,np.polyval(p1,x),'r-')
plt.plot(x,np.polyval(p2,x),'blue')
p4=np.polyfit(x,y,4)
plt.plot(x,np.polyval(p1,x),'r-')
plt.plot(x,np.polyval(p2,x),'blue')
plt.plot(x,np.polyval(p3,x),'green')
plt.plot(x,np.polyval(p4,x),'yellow')


# In[4]:


n=input("plz enter degree of polynomial")


# In[5]:


n=int(n)


# In[11]:


def polynomial_dataframe():
    for i in range(int(n)):
            pn=np.polyfit(x,y,n)
            plt.plot(x,np.polyval(pn,x))


# In[12]:


polynomial_dataframe()


# In[14]:


sales = pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week03\wk3_kc_house_train_data.csv', dtype=dtype_dict)


# In[20]:


sales = sales.sort_values(['sqft_living','price'], ascending=[True, False])


# In[ ]:




