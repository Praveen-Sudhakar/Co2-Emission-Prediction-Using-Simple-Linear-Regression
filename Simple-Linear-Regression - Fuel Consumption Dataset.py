#Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


#Reading the dataset & saving it in a variable.

df = pd.read_csv("D:\AIML\Dataset\FuelConsumption.csv")

df.head() #Reading the top 5 columns of the dataset


# In[5]:


#Understanding the size of the dataset 

df.shape


# In[6]:


#Understanding the details of the dataset

df.describe()


# In[7]:


#Creating a sub dataset out of the master dataset

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]


# In[8]:


cdf


# In[9]:


cdf[["ENGINESIZE"]]


# In[10]:


#Plotting the graph to undertand the linearity of the data

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,c='red')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[11]:


plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,c='Green')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[12]:


plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,c='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[13]:


#Splitting the sub dataset in 80:20 ratio. 80% in train & 20% is test

msk = np.random.rand(len(cdf)) <= 0.80


# In[14]:


train = cdf[msk]
test = cdf[~msk]


# In[15]:


train


# In[16]:


test


# In[17]:


#Plotting the train dataset to check the linearity

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,c='Orange')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[18]:


#Declaring the IV & DV & storing it in variables. Basically we are creating an array of train dataset & storing it in a variable

x = np.asanyarray(train[["ENGINESIZE"]]) #IV
y = np.asanyarray(train[["CO2EMISSIONS"]]) #DV


# In[19]:


x


# In[20]:


y


# In[21]:


#Importing necessary libraries to calculate the coefficient & Intercept using the formula y=mx+c

from sklearn import linear_model


# In[22]:


regr = linear_model.LinearRegression()


# In[23]:


train_x = np.asanyarray(train[["ENGINESIZE"]]) #IV
train_y = np.asanyarray(train[["CO2EMISSIONS"]]) #DV


# In[24]:


#After this step we will generate the coeffcient & Intercept of the train_x & train_y

regr.fit(train_x,train_y)


# In[25]:


print(f"The coefficient m = {regr.coef_}")


# In[26]:


print(f"The intercept c = {regr.intercept_}")


# In[27]:


#Plot the graph to check the best fit line

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,c='yellow')
plt.plot(train_x,regr.coef_*train_x+regr.intercept_) #Here we are using y=mx+c formula & using the coef & intercept values to draw the best fit line  
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[28]:


#Now we will use the test data to create a predictive model

test


# In[29]:


test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])


# In[30]:


predicted_y = regr.predict(test_x)


# In[31]:


plt.scatter(test.ENGINESIZE,test.CO2EMISSIONS,c='orange')
plt.plot(test_x,regr.coef_*test_x+regr.intercept_)
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[32]:


from sklearn.metrics import r2_score


# In[40]:


print(f"Mean Absolute Error: {np.mean(np.absolute(predicted_y-test_y))}")


# In[41]:


print(f"Mean Square Error: %.2f" % np.mean((predicted_y-test_y) ** 2))


# In[43]:


print(f"R-2 Score: {r2_score(test_y,predicted_y)*100} %")

