# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Importing the dataset
data = pd.read_csv('bank.csv',sep=';')
X = data.iloc[:,:-1]
y = data.iloc[:,16]

# Univariate analusis
sns.set(font_scale=1.5)
countplt = sns.countplot(x='y',data=data,palette='hls')

# Incoding y into zero and one
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['y'] = le.fit_transform(data['y'])
data['y'].value_counts()

# Analysis uning pvioting table
out_label = data.groupby('y')
out_label.agg(np.mean)
data.describe(include=['O'])
data[['job', 'y']].groupby(['job'], as_index=False).mean().sort_values(by='y', ascending=False)
data[['education','y']].groupby('education',as_index=False).mean().sort_values(by='y',ascending=False)
data[['marital','y']].groupby(['marital'],as_index=False).mean().sort_values(by='y',ascending=False)
data[['education','y']].groupby('education',as_index=False).mean().sort_values(by='y',ascending=False)
data[['default', 'y']].groupby(['default'], as_index=False).mean().sort_values(by='y', ascending=False)
data[['housing', 'y']].groupby(['housing'], as_index=False).mean().sort_values(by='y', ascending=False)
data[['contact', 'y']].groupby(['contact'], as_index=False).mean().sort_values(by='y', ascending=False)
data[['loan', 'y']].groupby(['loan'], as_index=False).mean().sort_values(by='y', ascending=False)
data[['poutcome', 'y']].groupby(['poutcome'], as_index=False).mean().sort_values(by='y', ascending=False)

# Creating dummy variable for categorical variable
cat_list = ['job','marital','education','default','housing','loan','contact','month','poutcome']
df1 = X[cat_list]
df1 = pd.get_dummies(df1,drop_first=True)
data = X.drop(columns = cat_list,axis=1)
X = data.join(df1)

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)

# Fitting Logistic Regression model to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
classifier.score(X_train,y_train)

# Prediction of test set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
classifier.score(X_test,y_test)