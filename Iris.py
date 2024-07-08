#!/usr/bin/env python
# coding: utf-8

# In[32]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[34]:


X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train.shape)
print(X_test.shape)


# In[54]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[55]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[57]:


sample = [[3,5,4,2],[2,3,5,4]]
predictions = knn.predict(sample)
pred_species = [iris.target_names[p] for p in predictions]
print('predictions: ', pred_species)


# In[63]:


import joblib

joblib.dump(knn, 'mlbrain.joblib')


# In[65]:


model = joblib.load('mlbrain.joblib')
model.predict(X_test)
sample = [[3,5,4,2],[2,3,5,4]]
predictions = model.predict(sample)
pred_species = [iris.target_names[p] for p in predictions]
print('predictions: ', pred_species)


# In[ ]:




