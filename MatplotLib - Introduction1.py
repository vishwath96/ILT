#!/usr/bin/env python
# coding: utf-8

# In[3]:


## Import matplotlib.pyplot
## !pip install matplotlib
## !conda install matplot


# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Line Plot

# In[3]:


x = np.arange(0, 10, 0.1)

y = 2*x + 5


# In[4]:


x


# In[5]:


y


# In[6]:


plt.plot(x, y)
plt.show()


# In[7]:


## Customize the line plot

x = np.arange(0, 10, 0.1)

y = 2*x + 5

fig = plt.figure(figsize=(10, 5))

plt.plot(x, y, linewidth=2.0, linestyle=":", color='m', alpha=0.5, marker='o')

plt.title("Line Plot Example")

plt.xlabel('X-Axis')

plt.ylabel("Y-Axis")

plt.legend(['Simple Line'], loc='right')

plt.show()


# In[10]:


## Customize the line plot

x = np.arange(0, 10, 1)

y = 2*x + 5

fig = plt.figure(figsize=(10, 5))

plt.plot(x, y, linewidth=2.0, linestyle=":", color='b', alpha=0.9, marker='^')

plt.title("Line Plot Example")

plt.xlabel('X-Axis')

plt.ylabel("Y-Axis")

plt.legend(['Simple Line'], loc='upper left')

plt.show()


# In[28]:


## Customize the line plot

x = np.arange(0, 10, 1)

y = 2*x + 5

y1 = 4.2*x + 1

fig = plt.figure(figsize=(10, 5))

plt.plot(x, y, linewidth=2.0, linestyle=":", color='b', alpha=0.9, marker='^')

plt.plot(x, y1, linewidth=2.0, linestyle=":", color='y', alpha=0.9, marker='o')

plt.title("Line Plot Example")

plt.xlabel('X-Axis')

plt.ylabel("Y-Axis")

plt.legend(['Sales of Suzuki Cars', 'Sales of Toyota Cars'], loc='best')

plt.grid(True)

plt.show()


# In[33]:


x = np.arange(0, 10, 1)
y1 = 2*x + 5
y2 = 3*x + 10

plt.subplot(1, 2, 1)
plt.plot(x, y1)
plt.title('Line plot for Y1')

plt.subplot(1, 2, 2)
plt.plot(x, y2)
plt.title("Line plot for Y2")

plt.show()


# In[35]:


x = np.arange(0, 10, 1)
y1 = 2*x + 5
y2 = 3*x + 10

plt.subplot(1, 3, 1)
plt.plot(x, y1)
plt.title('Line plot for Y1')

plt.subplot(1, 3, 3)
plt.plot(x, y2)
plt.title("Line plot for Y2")

plt.show()


# ## Creating plots from Pandas / External Data

# In[60]:


df = pd.DataFrame.from_dict(data={'Apples':20, 'Mangoes':15, 'Lemons':30, 'Oranges':10}, orient='index')
df.reset_index(inplace=True)
df.columns = ['Fruits', 'Sales']
df


# In[42]:


fig = plt.figure(figsize=(5, 2))

plt.plot(df['Fruits'], df['Sales'], linewidth=2.0, linestyle="--", color='b', alpha=0.9, marker='*')

plt.legend(['Sales of Fruits'], loc='best')

plt.show()


# ## How to create a bar Plot

# In[54]:


data = {'apples':20, 'mangoes':15, 'lemons':35, 'oranges':10}
names = list(data.keys())
sales = list(data.values())


# In[55]:


names


# In[56]:


sales


# In[59]:


fig = plt.figure(figsize=(10,3))

plt.bar(names, sales)

plt.show()


# ## Customizing Bar Chart

# In[61]:


data = {'apples':20, 'mangoes':15, 'lemons':35, 'oranges':10}
names = list(data.keys())
sales = list(data.values())


# In[66]:


fig = plt.figure(figsize=(10,5))

plt.barh(names, sales, color='yellow', alpha=0.9)

plt.title("Fruits Sales")

plt.xlabel("Sales")

plt.ylabel("Fruits")

plt.show()


# ## Scatter Plot

# In[67]:


a = [10, 20, 30, 40, 50, 60, 70]

b = [5, 3, 2, 9, 6, 8, 1]

fig = plt.figure(figsize=(10,5))

plt.scatter(a, b)

plt.show()


# In[78]:


## Customize the scatter plot

a = [10, 20, 30, 40, 50, 60, 70]

b = [5, 3, 2, 9, 6, 8, 1]

x = [1, 2, 3, 4, 5, 6, 7]

fig = plt.figure(figsize=(10,8))

plt.scatter(a, b, c='green', s=100, edgecolors='y', marker='o', alpha=0.8)

plt.scatter(a, x, c='yellow', s=20, edgecolors='b', marker='*', alpha=0.9)

plt.legend(['b', 'x'], loc='best')

plt.xlabel("X-axis")

plt.ylabel("Y-axis")

plt.show()


# In[81]:


## Customize the scatter plot

a = [10, 20, 30, 40, 50, 60, 70]

b = [5, 3, 2, 9, 6, 8, 1]

x = [1, 2, 3, 4, 5, 6, 7]

fig = plt.figure(figsize=(10,5))

plt.scatter(a, b, c='green', s=100, edgecolors='y', marker='o', alpha=0.8)

plt.scatter(a, x, c='yellow', s=20, edgecolors='b', marker='*', alpha=0.9)

plt.legend(['b', 'x'], loc='best')

plt.xlabel("X-axis")

plt.ylabel("Y-axis")

plt.grid(True)

plt.savefig("Scatter Plot Custom.png")

plt.show()


# In[84]:


## Saving and Read images

import matplotlib.pyplot as plt
image = plt.imread("Scatter Plot Custom.png")


# In[86]:


plt.imshow(image)


# ## Histogram

# In[87]:


numbers = [10, 12, 16, 19, 11, 20, 26, 28, 30, 38, 35, 34, 55, 45, 60, 62, 64, 70, 77, 78, 85, 94, 99]

plt.hist(numbers, bins = [0, 20, 40, 60, 80, 100])

plt.show()


# In[91]:


## Customizing histogram

numbers = [10, 12, 16, 19, 11, 20, 26, 28, 30, 38, 35, 34, 55, 45, 60, 62, 64, 70, 77, 78, 85, 94, 99]

plt.hist(numbers, bins = [0, 20, 40, 60, 80, 100], edgecolor='white', color='red', alpha=0.9)

plt.title("Histogram")

plt.xlabel("Steps in Range")

plt.ylabel("Number of Times We've achieved the steps")

# plt.grid(True, color='black')

plt.show()


# ## Box plot

# In[92]:


total = [20, 4, 1, 30, 20, 10, 20, 70, 30, 10]

orders = [10, 3, 1, 15, 17, 2, 30, 44, 2, 1]

discount = [30, 10, 20, 5, 10, 20, 50, 60, 20, 45]

data = list([total, orders, discount])


# In[94]:


fig = plt.figure(figsize=(10,7))

plt.boxplot(data)


plt.title("Box Plot on Sales")

plt.grid(True)

plt.show()


# In[ ]:


## Outlier == (q1 - 1.5*IQR) or (q3 + 15.*IQR)


# In[95]:


## Customize
total = [20, 4, 1, 30, 20, 10, 20, 70, 30, 10]

orders = [10, 3, 1, 15, 17, 2, 30, 44, 2, 1]

discount = [30, 10, 20, 5, 10, 20, 50, 60, 20, 45]

data = list([total, orders, discount])


fig = plt.figure(figsize=(10,7))

plt.boxplot(data, showmeans=True)


plt.title("Box Plot on Sales")

plt.grid(True)

plt.show()


# In[96]:


## Customize
total = [20, 4, 1, 30, 20, 10, 20, 70, 30, 10]

orders = [10, 3, 1, 15, 17, 2, 30, 44, 2, 1]

discount = [30, 10, 20, 5, 10, 20, 50, 60, 20, 45]

data = list([total, orders, discount])

fig = plt.figure(figsize=(10,7))

plt.boxplot(data, showmeans=True, meanline=True)

plt.title("Box Plot on Sales")

plt.grid(True)

plt.show()


# ## Violin Plot

# In[99]:


total = [20, 4, 1, 30, 20, 10, 20, 70, 30, 10]

orders = [10, 3, 1, 15, 17, 2, 30, 44, 2, 1]

discount = [30, 10, 20, 5, 10, 20, 50, 60, 20, 45]

data = list([total, orders, discount])

fig = plt.figure(figsize=(10,7))

plt.violinplot(data, showmedians=True, showmeans=True)

plt.title("Box Plot on Sales")

plt.grid(True)

plt.show()


# ## Create a pie chart

# In[3]:


Animals = ['Dog', 'Cat', 'Wolf', 'Lion']
Size = [50, 45, 60, 80]

plt.pie(Size, labels=Animals, autopct='%1.1f%%', shadow=True)

plt.title('Pie Chart Showing Animal Sizes')

plt.show()


# In[7]:


## Customizing Pie Chart
Animals = ['Dog', 'Cat', 'Wolf', 'Lion']
Size = [50, 45, 60, 80]

plt.pie(Size, labels=Animals, autopct='%.2f%%', shadow=True, startangle=45)

plt.title("Pie Chart Showing Animal Sizes, with a start angle of 45")

plt.show()


# In[8]:


## Customizing Pie Chart
Animals = ['Dog', 'Cat', 'Wolf', 'Lion']
Size = [50, 45, 60, 80]

plt.pie(Size, labels=Animals, autopct='%.2f%%', shadow=True, startangle=90)

plt.title("Pie Chart Showing Animal Sizes, with a start angle of 45")

plt.show()


# In[9]:


## String formatting in Python
name = 'mike'

age = 23

print(f"Hello, My name is {name} and I'm {age} years old.")


# In[22]:


Animals = ['Dog', 'Cat', 'Wolf', 'Lion']

Size = [50, 70, 90, 30]

plt.pie(Size, labels=Animals, autopct='%1.1f%%', shadow=True, startangle=60, explode=(x))

plt.title("Pie Chart with Explosion")

plt.show()


# In[14]:


max_ = max(Size)


# In[15]:


min_ = min(Size)


# In[21]:


x = [0 if i != max_ else 0.1 for i in Size]


# In[23]:


x


# ## Donut Chart

# In[42]:


group_names = ['GroupA', 'GroupB', 'GroupC']

group_size = [20, 30, 50]

size_centre = [5]

a, b, c = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

pie1 = plt.pie(group_size, labels=group_names, autopct='%1.1f%%', radius=1.5, pctdistance=0.6, explode=(0.05, 0.05, 0.05), colors=[a(0.5), b(0.5), c(0.8)])

pie2 = plt.pie(size_centre, radius = 0.5, colors='w')

plt.show()


# In[31]:


group_names = ['GroupA', 'GroupB', 'GroupC']

group_size = [20, 30, 50]

size_centre = [5, 10, 20]

a, b, c = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

pie1 = plt.pie(group_size, labels=group_names, radius=1.5, colors=[a(0.5), b(1.5), c(0.3)])

pie2 = plt.pie(size_centre, radius = 0.5, colors=[b(1.5),c(0.3), a(1.5)])

plt.show()


# ## Area Plot

# In[48]:


x = range(1, 15)

y = [1, 4, 6, 8, 4, 5, 3, 2, 4, 1, 5, 6, 8, 7]


# In[51]:


plt.stackplot(x, y, colors='b', alpha=0.5)

plt.plot(x, y, color='g')

plt.grid(True)

plt.show()


# In[52]:


a = range(0, 5)

y = [[1,2,3,4,5], [2,3,4,5,6], [7,8,9,1,2]]

fig = plt.figure(figsize=(10,5))

plt.stackplot(a, y, alpha=0.5)

plt.grid(True)

plt.title("Stack Plot")

plt.legend(['Area1', 'Area2', 'Area3'], loc='best')

plt.xlabel('X-Axis')

plt.ylabel('Y-Axis')

plt.show()


# ## Quiver Plot

# In[53]:


x_pos = 0

y_pos = 0

x_direction = 1

y_direction = 1

fig, ax = plt.subplots(figsize=(10, 5))

ax.quiver(x_pos, y_pos, x_direction, y_direction)

ax.set_title('Quiver Plot Showing Directions')

plt.show()


# ## Stream Plot

# In[54]:


x = np.arange(0, 10)

y = np.arange(0, 10)

X,Y = np.meshgrid(x, y)

u = np.ones((10, 10))

v = np.zeros((10, 10))

fig = plt.figure(figsize=(12,7))

plt.streamplot(X, Y, u, v)

plt.show()




# In[ ]:




