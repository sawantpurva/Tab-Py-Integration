
# coding: utf-8

# In[11]:


import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedKFold, cross_val_score, cross_val_predict 
from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score, classification_report

df =  pd.read_csv('/Users/Purva Sawant/breastcancer.csv', header=0)
df.drop(['Id'], 1, inplace=True)
encoder = preprocessing.LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])
X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# 10 fold stratified cross validation
kf = StratifiedKFold(y, n_folds=10, random_state=None, shuffle=True)

# Logistic regression with 10 fold stratified cross-validation using model specific cross-validation in scikit-learn
lgclf = LogisticRegressionCV(Cs=list(np.power(10.0, np.arange(-10, 10))),penalty='l2',scoring='roc_auc',cv=kf)
lgclf.fit(X, y)
y_pred = lgclf.predict(X)

# Show classification report for the best model (set of parameters) run over the full dataset
print("Classification report:")
print(classification_report(y, y_pred))

# Show accuracy and area under ROC curve
print("Accuracy: for Logistic %0.3f" % accuracy_score(y, y_pred, normalize=True))
print("Aucroc: %0.3f" % metrics.roc_auc_score(y, y_pred))

# Naive Bayes with 10 fold stratified cross-validation
nbclf = GaussianNB()
scores = cross_val_score(nbclf, X, y, cv=kf, scoring='roc_auc')

# Show accuracy statistics for cross-validation
print("Accuracy for Naive Bayes : %0.3f" % (scores.mean()))
print("Aucroc: %0.3f" % metrics.roc_auc_score(y, cross_val_predict(nbclf, X, y, cv=kf)))


# In[12]:


# The scoring function that will use the Naive Bayes Classifier to classify new data points
def SuggestDiagnosis(Cl_thickness, Cell_size, Cell_shape, Marg_adhesion, Epith_c_size, 
                     Bare_nuclei, Bl_cromatin, Normal_nucleoli, Mitoses):
    X = np.column_stack([Cl_thickness, Cell_size, Cell_shape, Marg_adhesion, Epith_c_size, 
                         Bare_nuclei, Bl_cromatin, Normal_nucleoli, Mitoses])
    X = scaler.transform(X)
    return encoder.inverse_transform(nbclf.predict(X)).tolist()


# In[8]:


import tabpy_client
connection = tabpy_client.Client('http://localhost:9004/')
connection.deploy('DiagnosticsDemo2',
                  SuggestDiagnosis,
                  'Returns diagnosis suggestion based on ensemble model trained using Wisconsin Breast Cancer dataset')

