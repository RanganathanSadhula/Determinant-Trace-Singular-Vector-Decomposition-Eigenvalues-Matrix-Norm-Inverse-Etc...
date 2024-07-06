#!/usr/bin/env python
# coding: utf-8

# ### Load in NumPy (remember to pip install numpy first)

# In[1]:


import numpy as np


# ### The Basics

# In[2]:


a = np.array([1,2,3], dtype='int32')
print(a)


# In[4]:


b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)


# In[5]:


# Get Dimension
a.ndim


# In[6]:


# Get Shape
b.shape


# In[8]:


# Get Type
a.dtype


# In[9]:


# Get Size
a.itemsize


# In[10]:


# Get total size
a.nbytes


# In[35]:


# Get number of elements
a.size


# ### Accessing/Changing specific elements, rows, columns, etc

# In[15]:


a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]]) 
print(a)


# In[16]:


# Get a specific element [r, c]
a[1, 5]


# In[17]:


# Get a specific row 
a[0, :]


# In[18]:


# Get a specific column
a[:, 2]


# In[20]:


# Getting a little more fancy [startindex:endindex:stepsize]
a[0, 1:-1:2]


# In[24]:


a[1,5] = 20

a[:,2] = [1,2]
print(a)


# *3-d example

# In[25]:






b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)


# In[30]:


# Get specific element (work outside in)
b[0,1,1]


# In[34]:


# replace 
b[:,1,:] = [[9,9,9],[8,8]]


# In[33]:


b


# ### Initializing Different Types of Arrays

# In[40]:


# All 0s matrix
np.zeros((2,3))


# In[42]:


# All 1s matrix
np.ones((4,2,2), dtype='int32')


# In[44]:


# Any other number
np.full((2,2), 99)


# In[49]:


# Any other number (full_like)
np.full_like(a, 4)


# In[56]:


# Random decimal numbers
np.random.rand(4,2)


# In[73]:


# Random Integer values
np.random.randint(-4,8, size=(3,3))


# In[76]:


# The identity matrix
np.identity(5)


# In[82]:


# Repeat an array
arr = np.array([[1,2,3]])
r1 = np.repeat(arr,3, axis=0)
print(r1)


# In[89]:


output = np.ones((5,5))
print(output)

z = np.zeros((3,3))
z[1,1] = 9
print(z)

output[1:-1,1:-1] = z
print(output)


# ##### Be careful when copying arrays!!!

# In[98]:


a = np.array([1,2,3])
b = a.copy()
b[0] = 100

print(a)


# ### Mathematics

# In[111]:


a = np.array([1,2,3,4])
print(a)


# In[109]:


a + 2


# In[102]:


a - 2


# In[103]:


a * 2


# In[104]:


a / 2


# In[118]:


b = np.array([1,0,1,0])
a + b


# In[113]:


a ** 2


# In[116]:


# Take the sin
np.cos(a)


# In[117]:


# For a lot more (https://docs.scipy.org/doc/numpy/reference/routines.math.html)


# ##### Linear Algebra

# In[127]:


a = np.ones((2,3))
print(a)

b = np.full((3,2), 2)
print(b)

np.matmul(a,b)


# In[132]:


# Find the determinant
c = np.identity(3)
np.linalg.det(c)


# In[133]:


## Reference docs (https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)

# Determinant
# Trace
# Singular Vector Decomposition
# Eigenvalues
# Matrix Norm
# Inverse
# Etc...


# ##### Statistics

# In[134]:


stats = np.array([[1,2,3],[4,5,6]])
stats


# In[139]:


np.min(stats)


# In[141]:


np.max(stats, axis=1)


# In[143]:


np.sum(stats, axis=0)


# ### Reorganizing Arrays

# In[151]:


before = np.array([[1,2,3,4],[5,6,7,8]])
print(before)

after = before.reshape((2,3))
print(after)


# In[158]:


# Vertically stacking vectors
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

np.vstack([v1,v2,v1,v2])


# In[164]:


# Horizontal  stack
h1 = np.ones((2,4))
h2 = np.zeros((2,2))

np.hstack((h1,h2))


# ### Miscellaneous
# ##### Load Data from File

# In[179]:


filedata = np.genfromtxt('data.txt', delimiter=',')
filedata = filedata.astype('int32')
print(filedata)


# ##### Boolean Masking and Advanced Indexing

# In[196]:


(~((filedata > 50) & (filedata < 100)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




