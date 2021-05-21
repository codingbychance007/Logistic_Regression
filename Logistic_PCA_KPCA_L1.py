'''My first model that i understand'''

'''Loading the required libraries'''
import pandas as pd
import matplotlib as mp
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

'''importing the data and check the overview'''
df = pd.read_csv("/Users/sky/Downloads/ML_Practice/Model Evaluation/wdbc.data", header=None)
df.head()

'''Dependent and Independent variable - labelencoding '''

X = df.loc[:, 2:]
y = df.loc[:, 1]

le = LabelEncoder()
y = le.fit_transform(y)
le.transform(['B','M'])

'''Splitting data into test and train'''

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1,stratify=y)

'''fitting a logistic regression model on the dataset - scaling, PCA, model using Pipeline operator'''

pipe_lr =make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1,solver='lbfgs'))
pipe_lr.fit(X_train,y_train)
y_pred=pipe_lr.predict(X_test)
print("Test accuracy : ", pipe_lr.score(X_test,y_test))

'''Standardization,KernelPCA,LogisticRegression'''
pipe_lr2 =make_pipeline(StandardScaler(),KernelPCA(n_components=2,kernel='sigmoid' ,gamma=30) ,LogisticRegression(random_state=1,solver='lbfgs'))
pipe_lr2.fit(X_train,y_train)
y_pred2=pipe_lr.predict(X_test)
print("Test accuracy : ", pipe_lr2.score(X_test,y_test))

'''Standardization,Logistic Regression with L1 regulatization'''
pipe_lr3 =make_pipeline(StandardScaler(),LogisticRegression(penalty='l1',random_state=1,solver='liblinear',C=2.0,multi_class='ovr'))
pipe_lr3.fit(X_train,y_train)
y_pred3=pipe_lr3.predict(X_test)
print("Test accuracy : ", pipe_lr3.score(X_test,y_test))