#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


Data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
Data[:5]


# In[3]:


Target = [(i+5)/100 for i in range(100)]
Target[:5]


# In[4]:


data = np.array(Data,dtype=float)
target = np.array(Target,dtype=float)


# In[5]:


data.shape, target.shape


# In[6]:


#Dividing data into train & test
x_train, x_test, y_train, y_test = train_test_split(data,target,test_size=0.2,random_state=4)


# In[7]:


#RNN 
model = Sequential()  


# In[8]:


model.add(LSTM((1),batch_input_shape=(None,5,1),return_sequences=False))


# In[9]:


model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])


# In[10]:


model.summary()


# In[11]:


history = model.fit(x_train,y_train,epochs=500,validation_data=(x_test,y_test))


# In[15]:


results = model.predict(x_test)


# In[17]:


plt.plot(history.history['loss'])
plt.show()


# In[ ]:




