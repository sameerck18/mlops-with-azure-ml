#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('wine-quality-data.csv') 
df


# In[2]:


# X will contain the data for 11 columns used for predicting.
X = df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']].values
# y is the traget column i.e., it has wine quality with scores from 0 to 10.
y = df['quality']


# In[3]:


# train_test_split library is used to split our data into train and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


#The random forest classifier is a supervised learning algorithm which you can use for regression and classification problems.
#n_estimators is the number of trees in the forest.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200) 
rfc.fit(X_train, y_train)


# In[5]:


#The confusion_matrix function evaluates classification accuracy by computing the confusion matrix with each row corresponding to the true class.
#The classification_report function builds a text report showing the main classification metrics.
from sklearn.metrics import confusion_matrix, classification_report
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# In[6]:


#The accuracy_score function computes the accuracy, either the fraction or the count of correct predictions.
from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_rfc)
cm


# In[7]:


df.head(10)


# In[8]:


Xnew = [[7.0,	0.27,	0.36,	20.7,	0.045,	45.0,	170.0,	1.0010,	3.00,	0.45,	8.8]]
ynew = rfc.predict(Xnew)
print('The quality of wine with given parameters is:') 
print(ynew)

