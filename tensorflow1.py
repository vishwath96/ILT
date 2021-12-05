#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[23]:


import numpy as np


# In[28]:


np.array([[1,2,3], [4,5,6]]).shape


# In[25]:


x = np.arange(24)
x = x.reshape(6,4)
x


# In[26]:


## Zero rank or scalar, contains no axes/dimensions
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)


# In[16]:


# Let's make this a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)


# In[18]:


# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)


# In[19]:


# There can be an arbitrary number of axes (sometimes called "dimensions")
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor)


# In[ ]:


## https://www.tensorflow.org/guide/tensor


# In[20]:


a = tf.constant(10)


# In[22]:




