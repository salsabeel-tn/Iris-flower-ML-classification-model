#!/usr/bin/env python
# coding: utf-8

# ##                 Iris flower classification ML model
# 

# In[28]:


import scipy 
import sklearn
import pandas as pd
import numpy as np


# In[3]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[4]:


print("keys of iris_dataset: \n{}".format(iris_dataset.keys()))


# In[5]:


#here I am diving my dataset into 75% as train data and 25% as test data for the ML classification module
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)


# In[14]:


# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
from pandas.plotting import scatter_matrix
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8)


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[17]:


#feeding data into the algorithm to construct the model
knn.fit(X_train,y_train) 


# In[18]:


X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))


# In[19]:


prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
iris_dataset['target_names'][prediction]))


# In[22]:


X_new1 = np.array([[3, 2, 5, 0.1]])
print("X_new.shape: {}".format(X_new.shape))


# In[23]:


prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
iris_dataset['target_names'][prediction]))


# In[24]:


#testing the model with the test data
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))


# In[25]:


#evaluating the model's accuracy against test data (1)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))


# In[26]:


#evaluating the model's accuracy against test data (2)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# * considerably high accuracy..
# * fit,predict and score functions are common functions in supervised ML models 

# In[ ]:




