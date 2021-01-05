import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

print(data_train.head())

#### from below code it show that their is high chance of survival for femail
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)
plt.show()

sns.barplot(x="Pclass", y="Survived", hue="Sex", data=data_train)
plt.show()

################## Preprocessing
## HAndling missing value

print(data_train.isnull().sum())

## Out of 891, 687 are the missing value in Cabin so we can drop that column
### Also Name and passnger ID is also not a usefull columns since they are unique column in data frame
data_train = data_train.drop(['Cabin','PassengerId','Name',"Embarked"], axis=1)
data_test = data_test.drop(['Cabin','PassengerId','Name',"Embarked"],axis = 1)

#we will replace the NA value in AGE column with the middle value of 2nd and 3rd Qurtile

data_train.Age = data_train.Age.fillna((data_train.Age.quantile(0.5)+data_train.Age.quantile(0.75))/2)
data_test.Age = data_test.Age.fillna((data_test.Age.quantile(0.5)+data_test.Age.quantile(0.75))/2)

obj_type = list(data_train.select_dtypes(['object']).columns)

#### Now we will encode the column with data type as String or object

from sklearn import preprocessing

def encode_features(df_train, df_test):
    features = ['Sex', 'Ticket']
    
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

data_train, data_test = encode_features(data_train, data_test)

#############  Now we will traine a model
from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


### import the class
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


#### we got Accuracy score 0.787 using random forest

###  Now lets try Logistic regression
print("Logistic Regresion")
model = LogisticRegression()

# Choose some parameter combinations to try
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'max_iter': list(range(100,800,100)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

grid_search = GridSearchCV(model, param_grid=param_grid, refit = True, verbose = 1, cv=5)

grid_obj = grid_search.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf_logistic = grid_search.best_estimator_

# Fit the best algorithm to the data. 
clf_logistic.fit(X_train, y_train)

predictions = clf_logistic.predict(X_test)
print(accuracy_score(y_test, predictions))
