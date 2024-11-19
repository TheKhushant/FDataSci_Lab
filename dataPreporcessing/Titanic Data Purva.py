#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[3]:


df = pd.read_csv("titanic.csv")
df


# In[14]:


df.head()


# In[9]:


df.tail()


# In[15]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.isnull()


# In[23]:


df["Age"].isnull()


# In[24]:


df["Age"].mean()


# In[55]:


df["Age"] = df["Age"].fillna(df["Age"].mean())


# In[56]:


df["Age"].isnull()


# In[57]:


df["Age"].mode()


# In[58]:


df["Age"].median()


# In[59]:


df.drop(columns="Cabin")


# In[60]:


df.info()


# In[61]:


df.Sex.unique


# In[62]:


df.Sex.str.upper()


# In[63]:


le = LabelEncoder()
le.fit_transform(df['Sex'])


# In[64]:


df.info()


# In[65]:


df.Embarked.unique()


# In[66]:


df['Embarked'] = df.Embarked.fillna(df.Embarked.mode()[0])


# In[67]:


df.info()


# In[68]:


df.shape


# In[69]:


from sklearn.preprocessing import OneHotEncoder


# In[70]:


df_new = pd.get_dummies(df, columns = ['Embarked'])


# In[71]:


df_new.shape


# In[72]:


df_new.info()


# In[73]:


df_new.head()


# In[74]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[83]:


plt.boxplot(df.Age, vert = False)
plt.show()


# In[75]:


sns.boxplot(df.Age)


# In[76]:


bins = [0,18,35,60,100]
labels = ["Child","Young","Adult","Senior Citizen"]


# In[77]:


df["Age_Group"] = pd.cut(df.Age, bins = bins, labels = labels)


# In[78]:


df.info()


# In[79]:


df.Age_Group


# In[80]:


df.Age_Group.unique()


# In[81]:


df.Age_Group.nunique()


# In[1]:


df.shape


# In[2]:


df.shape


# In[ ]:




