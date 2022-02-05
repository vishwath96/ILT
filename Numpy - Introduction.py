#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Array

a1 = ['hi', 3, 0.5]


# In[2]:


print(a1)


# In[3]:


type(a1)


# ## Installing and Importing Numpy

# In[4]:


## https://github.com/numpy/numpy -- documentation for understanding the library better


# In[5]:


## !conda install numpy --y


# In[6]:


## !pip install numpy


## Mac, python 3.8, conda as my environment


# In[7]:


import numpy as np


# ## Creating a simple NumPy array using np.array

# In[8]:


print(a1)


# In[9]:


numpy_array = np.array([1,2,3])
print(numpy_array)


# In[10]:


type(numpy_array)


# In[11]:


type(numpy_array[0])


# ## Types of array and visualization

# In[12]:


## 1D, 2D, 3D and ND -- types of arrays


# In[13]:


a = np.array([1,2,3])
print(a)


# In[14]:


a.shape


# In[15]:


b = np.array([[1,2], [4,5]])
print(b)


# In[16]:


b.shape


# In[17]:


c = np.array([[[1,2,3], [4,5,6], [7,8,9]]])
print(c)


# In[18]:


c.shape


# ## Array Initializations

# In[19]:


zeros = np.zeros((1,2))
print(zeros)


# In[20]:


zeros.shape


# In[21]:


zeros1 = np.zeros(5)
zeros1


# In[22]:


zeros1.shape


# In[23]:


ones = np.ones((1,2))
print(ones)


# In[24]:


ones.shape


# ## Arrange numbers (creating arrays)

# In[25]:


## 0:5
a = np.arange(5)
a


# In[26]:


b = np.arange(0, 20, 2)
b


# In[27]:


c = np.arange(-1, -10, -2)
c


# ## Arranging 'z' numbers between 2 values

# In[28]:


lin_space = np.linspace(0, 10, 6)
lin_space


# In[29]:


lin_space1 = np.linspace(10, 100, 10)
lin_space1


# ## Creating Array of N Dimension using Same Numbers

# In[30]:


fill_array = np.full((2,2), 5)
fill_array


# ## Creating Array of N Dimension using Some Random Numbers

# In[31]:


random = np.random.random((2,2))
random


# In[32]:


randint = np.random.randint(low=0, high=10.5, size=(15,2))


# In[33]:


randint


# ## Finding a shape of an array

# In[34]:


randint


# In[35]:


randint.shape


# ## Reshaping an array

# In[36]:


a = np.arange(24)


# In[37]:


a


# In[38]:


a.shape


# In[39]:


a.shape = (12,2)


# In[40]:


b = a.reshape(12,2)


# In[41]:


b


# In[42]:


b = a.reshape(6,2,2)


# In[43]:


a.shape


# In[44]:


b = a.reshape(6,4)


# In[45]:


b


# ## Find size of a given array

# In[46]:


python_list = [1,2,3]

len(python_list)


# In[47]:


python_list1 = [[1,2,3], [4,5,6]]
len(python_list1)


# In[48]:


python_list1


# In[49]:


a = np.arange(24)


# In[50]:


a


# In[51]:


b = a.reshape(12,2)


# In[52]:


b


# In[53]:


b.size


# ## Finding Dimension of an array

# In[54]:


b


# In[55]:


b.ndim


# ## Addition of NumPy Arrays

# In[56]:


x = np.sum([[1,2], [2,3]])
x


# In[57]:


y = np.sum([[1,2], [2,3]], axis=0)
y


# In[58]:


y = np.sum([[3,2], [2,3]], axis=1)
y


# ## Subtraction of Numpy Arrays

# In[59]:


a = np.array([3,2])
b = np.array([1,2])


# In[60]:


np.subtract(a,b)


# ## Division of NumPy Arrays

# In[61]:


a = np.array([3,2])
b = np.array([1,2])


# In[62]:


np.divide(a,b)


# ## Multiplication of NumPy Arrays

# In[63]:


a = np.array([3,2])
b = np.array([1,2])


# In[64]:


np.multiply(a,b)


# ## Exponential Operations

# In[65]:


a = np.array([1,1])

print(np.exp(a))


# In[66]:


## e -- euler's number (log base)


# ## Square Root Operations

# In[67]:


a = np.arange(3)
a


# In[68]:


np.sqrt(a)


# ## Trignometric Operations

# In[69]:


## sin & cos and Log
print(np.sin(a))
print(np.cos(a))
print(np.log(a))


# ## Element wise Array Comparison

# In[70]:


a = np.arange(6)
b = np.array([0, 1, 2, 3, 4, 6])


# In[71]:


print(a)
print(b)


# In[72]:


np.equal(a,b)


# ## Array Wise Comparison

# In[73]:


np.array_equal(a,b)


# ## Aggregate Functions

# In[74]:


a = np.array([1,2,3,4,5])


# In[75]:


## Mean or average
a.mean()


# In[76]:


## Sum
a.sum()


# In[78]:


## Standard Deviation
# np.std(a)

# SquareRoot(Summation(x-mean)^2)/ N = 5
a.std()


# In[79]:


## correlation co-efficient
x = np.random.random((2,2))
x


# In[80]:


np.corrcoef(x)


# ## Broadcasting

# In[81]:


## [1,2,3] + 4
a = np.array([1, 2, 3])
a.shape


# In[82]:


b = np.array([4,5,6,7])


# In[83]:


a + b


# In[84]:


## 2D Array

a = np.array([[1,2,3], [4,5,6]])
b = 3


# In[85]:


a


# ## Indexing

# In[86]:


a = ['a', 'b', 'c', 'd', 'e']
a


# In[87]:


a[2]


# ## Slicing

# In[88]:


a[1:]

## first element would be inclusive
## last element would be exclusive


# In[89]:


a[1:4]


# In[90]:


a[:-1]


# In[91]:


'''a , b, c'''

a[-5:-2]


# In[92]:


a[:-3]


# ## Indexing and Slicing on 2D Array

# In[93]:


a = np.array([[1,2,3], [4,5,6], [7,8,9]])


# In[94]:


a.shape


# In[95]:


a


# In[96]:


a[1][1]


# In[97]:


a[1,1]


# In[98]:


a[:2,:1]


# ## Array Concatentation

# In[99]:


a = np.array([1,2,3])
b = np.array([4,5,6])

np.concatenate((a,b))


# ## Stacking

# In[100]:


## Vertical Stack
c = np.vstack((a,b))


# In[101]:


c


# In[102]:


c.shape


# In[103]:


## Horizontal Stack
d = np.hstack((a,b))
d


# In[104]:


d.shape


# ## Combining Column Wise - Stacking

# In[105]:


e = np.column_stack((a,b))


# In[106]:


e


# In[107]:


e.shape


# ## Splitting Arrays

# In[108]:


x = np.arange(16)


# In[109]:


x


# In[110]:


y = x.reshape(4,4)


# In[111]:


y


# In[112]:


z = np.hsplit(y, 2)


# In[113]:


z


# In[120]:


zz = np.vsplit(y, 2)


# In[122]:


zz


# ## Advantages of NumPy Arrays over Python Lists

# In[123]:


## 1. Faster
## 2. Less memory


# In[124]:


import sys


# In[125]:


py_array = range(1000)

print("Size of each element in the python list of py_array in bytes: ", sys.getsizeof(py_array))


# In[126]:


print("Size of entire python list of py_array in bytes is: ", sys.getsizeof(py_array)*len(py_array))


# In[127]:


np_array = np.arange(1000)

print("Size of each element in numpy array is: ", np_array.itemsize)


# In[128]:


## Speed


# In[129]:


import time


# In[130]:


list_1 = range(1000)
list_2 = range(1000)


np_array1 = np.arange(1000)
np_array2 = np.arange(1000)


initial_time = time.time()

resultant_list = [(a*b) for a, b in zip(list_1, list_2)]

print("Total time taken for the operation is: ", time.time() - initial_time, "seconds")


# In[131]:


initial_time = time.time()

resultant_array = np_array1 * np_array2

print("Total time taken for the operation is: ", time.time() - initial_time, "seconds")


# In[ ]:





# In[ ]:




