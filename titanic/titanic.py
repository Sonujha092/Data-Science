# Titanic 

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random as rnd

# Importingt the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df,test_df]

# Analyze by describing data
train_df.columns
''' categorical features : Survived,Sex,Embarked
    ordinal : Pclass
    continuous : Age,Fare
    Descrete : Sibsb,Parch
'''
# Preview the data
train_df.head()
train_df.tail()

train_df.info()
test_df.info()
'''Age,Cabin and Embarked has missing values in training set
    Age,Cabin has missing value in test set 
'''
train_df.describe()
test_df.describe()

# Destribution of categorical variable
train_df.describe(include=['O'])

# Analysing by pvioting features
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'],as_index = False).mean().sort_values(by='Survived',ascending = False )
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch','Survived']].groupby(['Parch'],as_index = False).mean().sort_values(by='Survived',ascending = False)

# Analysing by Visualizing data
g = sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Age',bins = 20)
'''Infants (Age <=4) had high survival rate.
    Oldest passengers (Age = 80) survived.
    Large number of 15-25 year olds did not survive.
    Most passengers are in 15-35 age range.
'''
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df,col='Survived',row='Pclass',size = 2.2,aspect=1.6)
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend()
'''Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
    Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
    Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
    Pclass varies in terms of Age distribution of passengers.
'''

# Correlating categorical features
# grid = sns.FacetGrid(train_df,col='Embarked)
grid = sns.FacetGrid(train_df,col='Embarked')
grid.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')
grid.add_legend()
'''Female passengers had much better survival rate than males.
    Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
    Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports.
    Ports of embarkation have varying survival rates for Pclass=3 and among male passengers.
'''
# Correlating categorical and numerical features
grid = sns.FacetGrid(train_df,row ='Embarked',col='Survived',size=2.2,palette={0:'k',1:'w'})
grid.map(sns.barplot,'Sex','Fare',alpha=.5,ci=None)
grid.add_legend()
'''Higher fare paying passengers had better survival.
    Port of embarkation correlates with survival rates.
'''
# Dropping columns
train_df = train_df.drop(['Ticket','Cabin'],axis=1)
test_df=test_df.drop(['Ticket','Cabin'],axis=1)
combine = [train_df,test_df]

#
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

# training and test set
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

# Taking care of categorical variable
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
