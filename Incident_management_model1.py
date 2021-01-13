# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:48:08 2020

@author: Chinmay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('E:\\Project1\\Incidents_service.csv')

###################EDA##########################
df.head
df.shape
df=pd.merge(df,df.groupby(['ID','ID_status'])['updated_at'].max(),on=['ID','ID_status','updated_at'])
drop=['opened_time','updated_at','created_at','created_at']
df.drop(drop,axis=1,inplace=True)
df.head
df.columns
df.shape
#droping problem_id and change request as more than 98% of the value as '?'
drop1=['problem_id','change request']
df.drop(drop1,axis=1,inplace=True)
df.dtypes
df.columns


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import classification_report

X=df[[ 'ID','ID_status', 'active', 'count_reassign', 'count_opening',
       'count_updated', 'ID_caller', 'opened_by',  'Created_by',
       'updated_by', 'type_contact', 'location',
       'category_ID', 'user_symptom',  'Support_group',
       'support_incharge']]

y=df[['impact']]


####converting into categorical data####
categorical_features_indices=np.where(X.dtypes!=np.float)[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=3)
y_train.impact.value_counts()
y_test.impact.value_counts()
X_train.columns
df.columns


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

#Model building---------------------------------------------
kf=StratifiedKFold(n_splits=5)
for train_index,test_index in kf.split(X_train,y_train):
    X_train1,X_test1,y_train1,y_test1=X_train.iloc[train_index],X_train.iloc[test_index],y_train.iloc[train_index],y_train.iloc[test_index]
    clf=CatBoostClassifier(n_estimators=500)
    clf.fit(X_train,y_train,cat_features=categorical_features_indices)

#######prediction#######
pred=clf.predict(X_test)
print(classification_report(pred,y_test))

pred1=clf.predict(X_train)
print(classification_report(pred1,y_train))


from sklearn.metrics import f1_score
score=f1_score(pred,y_test,average='micro')
score

confusion_matrix(y_test,pred)

###############heatmap########
sns.heatmap(confusion_matrix(y_test,pred),annot=True,cmap='Blues',xticklabels=['1 - High', '2 - Medium ','3 - Low'],yticklabels=['1 - High', '2 - Medium ','3 - Low'],fmt='g')

############feature importance###########
clf.feature_importances_

plt.figure(figsize=(18,7))
for i in range(len(clf.feature_importances_)):
    print('Feature %d: %f' % (i, clf.feature_importances_[i]))
# plot the scores
plt.bar([i for i in X.columns], clf.feature_importances_)
plt.show()

#saving model to pickle file
import pickle
pickle.dump(clf,open('Incident_management_model1.pkl','wb'))


















