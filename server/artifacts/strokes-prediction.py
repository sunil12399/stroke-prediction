#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
sklearn.__version__


# In[2]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from sklearn.feature_selection import mutual_info_classif


# In[3]:


def scores_print(y_true, predictions):
    print(f'accuracy = {accuracy_score(y_true, predictions)}')
    print(f'Cross-entropy = {log_loss(y_true, predictions)}')
    print(f'Confusion_matrix = \n{confusion_matrix(y_true, predictions)}')


# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[5]:


path = "healthcare-dataset-stroke-data.csv"


# In[6]:


df = pd.read_csv(path)
df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


data = df.copy()


# In[10]:


# data.bmi.fillna(data.bmi.median(), inplace = True)
data.dropna(inplace = True)
data = data[data['gender']!="Other"]


# In[11]:


data['age/bmi'] = data['age']/data['bmi']
data['age*bmi'] = data['age']*data['bmi']


# In[12]:


data.shape


# In[15]:


X, y = data.drop(['id', 'stroke'], axis = 1), data.stroke


# In[16]:


X_oh = pd.get_dummies(X)


# In[17]:


from imblearn.over_sampling import BorderlineSMOTE
sm = BorderlineSMOTE(random_state=123)
X_sm , y_sm = sm.fit_resample(X_oh,y)


# In[18]:


X_train, X_val, y_train, y_val =  train_test_split(X_sm, y_sm, stratify = y_sm, random_state = 777)


# In[19]:


scaler = StandardScaler()
histgradient = HistGradientBoostingClassifier(random_state=0)
logistic = LogisticRegression(random_state=0)
randomforest = RandomForestClassifier(random_state = 0)


# In[20]:


from imblearn.pipeline import Pipeline
histgradientmodel = Pipeline(
    steps = [
        ("scaler", scaler),
        ("classifier", histgradient)
    ]
)
cv_results = cross_validate(
    histgradientmodel, X_train, y_train, scoring="balanced_accuracy",
    return_train_score=True, return_estimator=True,
    n_jobs=-1
)
print(
    f"Balanced accuracy mean +/- std. dev.: "
    f"{cv_results['test_score'].mean():.3f} +/- "
    f"{cv_results['test_score'].std():.3f}"
)

scores = []
for fold_id, cv_model in enumerate(cv_results["estimator"]):
    scores.append(
        balanced_accuracy_score(
            y_val, cv_model.predict(X_val)
        )
    )
print(
    f"Balanced accuracy mean +/- std. dev.: "
    f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
)


# In[21]:


logisticmodel = Pipeline(
    steps = [
        ("scaler", scaler),
        ("classifier", logistic)
    ]
)
cv_results = cross_validate(
    logisticmodel, X_train, y_train, scoring="balanced_accuracy",
    return_train_score=True, return_estimator=True,
    n_jobs=-1
)
print(
    f"Balanced accuracy mean +/- std. dev.: "
    f"{cv_results['test_score'].mean():.3f} +/- "
    f"{cv_results['test_score'].std():.3f}"
)

scores = []
for fold_id, cv_model in enumerate(cv_results["estimator"]):
    scores.append(
        balanced_accuracy_score(
            y_val, cv_model.predict(X_val)
        )
    )
print(
    f"Balanced accuracy mean +/- std. dev.: "
    f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
)


# In[22]:


RFGmodel = Pipeline(
    steps = [
        ("scaler", scaler),
        ("classifier", randomforest)
    ]
)
cv_results = cross_validate(
    RFGmodel, X_train, y_train, scoring="balanced_accuracy",
    return_train_score=True, return_estimator=True,
    n_jobs=-1
)
print(
    f"Balanced accuracy mean +/- std. dev.: "
    f"{cv_results['test_score'].mean():.3f} +/- "
    f"{cv_results['test_score'].std():.3f}"
)

scores = []
for fold_id, cv_model in enumerate(cv_results["estimator"]):
    scores.append(
        balanced_accuracy_score(
            y_val, cv_model.predict(X_val)
        )
    )
print(
    f"Balanced accuracy mean +/- std. dev.: "
    f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
)


# In[23]:


logisticmodel.fit(X_train, y_train)
y_preds = logisticmodel.predict(X_val)
print()
scores_print(y_val, y_preds)


# In[24]:


RFGmodel.fit(X_train, y_train)
y_preds = RFGmodel.predict(X_val)
scores_print(y_val, y_preds)


# In[25]:


histgradientmodel.fit(X_train, y_train)
y_preds = histgradientmodel.predict(X_val)
scores_print(y_val, y_preds)


# In[26]:


from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators = [
    ('logistic', logisticmodel), ('random forest', RFGmodel), ('histgradient', histgradientmodel)
], voting="hard")
vc.fit(X_train, y_train)
y_preds = vc.predict(X_val)
scores_print(y_val, y_preds)


# In[27]:


import pickle
with open('Stroke_Prediction.pickle', 'wb') as  f:
    pickle.dump(vc, f)


# In[28]:


import json
columns = {
    'data_columns': [col.lower() for col in X_oh.columns]
}
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))


# In[ ]:




