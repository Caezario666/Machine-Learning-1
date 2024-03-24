#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Salary_Data.csv")


# In[3]:


df.head()


# In[6]:


X = df.iloc[:, :-1]


# In[7]:


y = df.iloc[:, 1]


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


regressor=LinearRegression()


# In[12]:


regressor.fit(X_train, y_train)


# In[13]:


y_pred = regressor.predict(X_test)


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


plt.scatter(X_train, y_train, color="green")


# In[16]:


plt.plot(X_train, regressor.predict(X_train), color="red")


# In[17]:


plt.title("Years Experience VS Salary")


# In[18]:


plt.xlabel("Years Experience")


# In[19]:


plt.ylabel("Salary")


# In[21]:


plt.show()


# In[22]:


plt.scatter(X_test, y_test, color="green")


# In[23]:


plt.scatter(X_test, y_test, color="green")


# In[24]:


plt.title("Years Experience VS Salary")


# In[25]:


plt.xlabel("Years Experience")


# In[26]:


plt.ylabel("Salary")


# In[27]:


plt.show()


# In[28]:


salary_pred = regressor.predict([[12]])


# In[29]:


print("The salary for that amount of years experience is: ", salary_pred)

