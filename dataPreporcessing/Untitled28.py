#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[11]:


dataset = pd.read_csv('dataset.csv')
print(dataset.shape)
print(dataset.head(5))


# In[12]:


dataset.info()


# In[13]:


transactions = []
for i in range(0, 7500):
    transactions.append([str(dataset.values[i,j])for j in range(0, 20)])
transactions


# In[16]:


get_ipython().system('pip install apyori')


# In[19]:


from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)


# In[20]:


results = list(rules)
results

