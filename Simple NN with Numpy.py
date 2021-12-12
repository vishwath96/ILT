#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[7]:


#  y = w1x1 + w2x2


# In[8]:


x1 = 2
x2 = 5
y = 31


# In[9]:


lr = 0.01
w1 = 3
w2 = 7
for epoch in range(50):
    y_pred = w1*x1 + w2*x2
    error = (y - y_pred)**2
    dEw1 = 2*(y - y_pred)*(-x1)
    dEw2 = 2*(y - y_pred)*(-x2)

    w1 = w1 - lr*dEw1
    w2 = w2 - lr*dEw2
    print("Value of w1 - ", w1, "Value of w2 - ", w2, "Error is - ", error)


# In[10]:


print(w1, w2)


# In[11]:


x1*w1 + x2*w2


# In[ ]:




