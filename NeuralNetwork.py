#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
"""
This code imports the necessary libraries for working with neural networks.

- `numpy` is imported as `np` for numerical computations.
- `matplotlib.pyplot` is imported as `plt` for data visualization.
"""
import matplotlib.pyplot as plt


# **INITIALISATION**

# ![Alt text](https://raw.githubusercontent.com/Susanta2102/Neural-Network-from-scratch/main/pic%201.jpeg)
# 

# In[23]:


def initialise(in_neuron,hidden_neuron,out_neuron):
  """
  Initializes the weights and biases for a neural network.

  Parameters:
  - in_neuron (int): Number of input neurons.
  - hidden_neuron (int): Number of hidden neurons.
  - out_neuron (int): Number of output neurons.

  Returns:
  - w1 (ndarray): Weight matrix for the connections between input and hidden layer.
  - b1 (ndarray): Bias vector for the hidden layer.
  - w2 (ndarray): Weight matrix for the connections between hidden and output layer.
  - b2 (ndarray): Bias vector for the output layer.
  """
  w1=np.random.rand(hidden_neuron,in_neuron)-.5
  w2=np.random.rand(out_neuron,hidden_neuron)-.5
  b1=np.zeros((hidden_neuron,1))#np.random.rand(hidden_neuron,1)
  b2=np.zeros((out_neuron,1))
  return w1,b1,w2,b2


# 

# In[27]:


def ReLU(z):
  return np.maximum(0,z)
def linear(z):
  return z
def linearPrime(z):
  return 1
def ReLUPrime(z):
  a=ReLU(z)
  return a>0


# ![Alt text](https://raw.githubusercontent.com/Susanta2102/Neural-Network-from-scratch/main/pic%202.jpeg)
# 

# In[26]:


def forwardProp(a0,w1,b1,w2,b2):
  z1=np.dot(w1,a0)+b1
  a1=ReLU(z1)
  z2=np.dot(w2,a1)+b2
  a2=linear(z2)
  return z1,a1,z2,a2



# ![Alt text](https://raw.githubusercontent.com/Susanta2102/Neural-Network-from-scratch/main/pic%203.jpeg)
# 

# In[6]:


def computeGradient(a2,y,z2,a1,w2,z1,a0):
  de2=(a2-y)*linearPrime(z2)
  db2=de2
  dw2=np.dot(de2,a1.T)
  da1=np.dot(w2.T,de2)
  de1=da1*ReLUPrime(z1)
  db1=de1
  dw1=np.dot(de1,a0.T)
  return dw1,db1,dw2,db2


# In[7]:


a0=np.array([0.5,0.5]).reshape(2,1)
y=np.array([-0.5,0.5]).reshape(2,1)
w1,b1,w2,b2=initialise(2,3,2)
z1,a1,z2,a2=forwardProp(a0,w1,b1,w2,b2)
dw1,db1,dw2,db2=computeGradient(a2,y,z2,a1,w2,z1,a0)
print(dw1)
print(db1)
print(dw2)
print(db2)


# ![Alt text](https://raw.githubusercontent.com/Susanta2102/Neural-Network-from-scratch/main/pic%204.jpeg)
# 

# In[8]:


def updateWeightsAndBiases(w1,b1,w2,b2,dw1,db1,dw2,db2,eta):
  w1=w1-eta*dw1
  b1=b1-eta*db1
  w2=w2-eta*dw2
  b2=b2-eta*db2
  return w1,b1,w2,b2


# In[9]:


eta=0.1
epoch=100

a0=np.array([0.5,0.5]).reshape(2,1)
y=np.array([-0.5,0.5]).reshape(2,1)

w1,b1,w2,b2=initialise(2,10,2)

for i in range(epoch):
  z1,a1,z2,a2=forwardProp(a0,w1,b1,w2,b2)
  dw1,db1,dw2,db2=computeGradient(a2,y,z2,a1,w2,z1,a0)
  w1,b1,w2,b2=updateWeightsAndBiases(w1,b1,w2,b2,dw1,db1,dw2,db2,eta)
  cost=np.sum((a2-y)**2)*.5
  print(a2)
  print(cost)


# In[10]:


x_input=np.random.uniform(-1,1,(1000,2))
print(x_input)
rm=[[-1,0],
    [0,1]]
x_ground=np.dot(rm,x_input.T).T
print(x_ground)


# In[11]:


plt.scatter(x_input[:,0],x_input[:,1],c='RED')
plt.scatter(x_ground[:,0],x_ground[:,1],c='GREEN')
plt.show()
     


# In[12]:


eta=0.1
epoch=100

a0=np.array([0.5,0.5]).reshape(2,1)
y=np.array([-0.5,0.5]).reshape(2,1)

w1,b1,w2,b2=initialise(2,10,2)

for i in range(epoch):
  cost=0;
  for j in range(1000):
    a0=x_input[j,:].T.reshape(2,1)
    y=x_ground[j,:].T.reshape(2,1)
    z1,a1,z2,a2=forwardProp(a0,w1,b1,w2,b2)
    dw1,db1,dw2,db2=computeGradient(a2,y,z2,a1,w2,z1,a0)
    w1,b1,w2,b2=updateWeightsAndBiases(w1,b1,w2,b2,dw1,db1,dw2,db2,eta)
    cost=cost+np.sum((a2-y)**2)*.5
  print(cost)


# In[13]:


test_x=np.arange(0,1,.01)
test_y=test_x**2

test=np.column_stack([test_x,test_y])
print(test)
     


# In[14]:


plt.scatter(test_x,test_y)
plt.show()


# In[15]:


n=test.shape[0]
o_x=[]
o_y=[]
for i in range(n):
  a0=test[i,:].T.reshape(2,1)
  z1,a1,z2,a2=forwardProp(a0,w1,b1,w2,b2)
  #print(a2)
  o_x.append(a2[0,0])
  o_y.append(a2[1,0])


# In[16]:


plt.scatter(o_x,o_y,c="blue")
plt.scatter(test[:,0],test[:,1],c="red")
plt.show()


# In[17]:


import pandas as pd
xtest = pd.read_csv('file.csv')
xtest=np.array(xtest)


# In[18]:


print(xtest.shape)
plt.scatter(xtest[:,0],xtest[:,1],s=.1,c='RED')


# In[19]:


n=xtest.shape[0]
o_x=[]
o_y=[]
for i in range(n):
  a0=xtest[i,:].T.reshape(2,1)
  z1,a1,z2,a2=forwardProp(a0,w1,b1,w2,b2)
  #print(a2)
  o_x.append(a2[0,0])
  o_y.append(a2[1,0])


# In[20]:


plt.scatter(o_x,o_y,s=.1,c="blue")
plt.scatter(xtest[:,0],xtest[:,1],s=.1,c="red")
plt.show()
     


# In[ ]:




