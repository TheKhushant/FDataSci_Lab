
# coding: utf-8

# In[4]:


#pearson program (without ploting it in graph)
import numpy as np
from scipy.stats import pearsonr
#create a data
temperature = np.array([20,22,25,27,30,35,40])
ics = np.array([200,220,260,270,300,350,400])
correlation, _ = pearsonr(temperature, ics)
print(f"Pearsonr correlation Coefficient (r) : {correlation:.3f}")

