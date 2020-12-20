import pandas as pd
import numpy as np
import random
import math

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape)
print(test.shape)

#cabin, ticket값 삭제
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
#embarked값 가공
train = train.fillna({"Embarked": "S"})
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
#name값 가공
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
#name, passengerId 삭제
train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]
#sex값 가공
sex_mapping = {"male": 0, "female": 1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
#age값 가공
train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels=labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels=labels)
#age값 추측해 넣기
age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
#ageGroup을 숫자로, age삭제
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)
#fare값 가공
'''train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])
train['FareBand'].astype(int)'''
train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)

'''train['Fare'].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test['Fare'].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train:
    print(dataset)
    dataset.loc[dataset['Fare']<=17, 'Fare']=0,
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30), 'Fare']=1,
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100), 'Fare']=2,
    dataset.loc[dataset['Fare']>100, 'Fare']=3
for dataset in test:
    dataset.loc[dataset['Fare']<=17, 'Fare']=0,
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30), 'Fare']=1,
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100), 'Fare']=2,
    dataset.loc[dataset['Fare']>100, 'Fare']=3'''
print(train.head())
print(test.head())
test_one = test
X = train.drop('Survived', axis=1)
Y = train['Survived']
T = test.drop('PassengerId', axis=1)
#print(X.head())
#print(Y.head())
#print(x.shape, y.shape)
#print(train.info())

#w초기화(9개)
w = [0, 0, 0, 0, 0, 0, 0, 0]
for i in range(0, 8):
    w[i] = random.uniform(-1.0, 1.0)
    print(w[i])

par = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'AgeGroup']
a = 0.001

def ztrain(n):
    j = 0
    for i in range(0, 8):
        if i == 0:
            j = j + w[i]
        else:
            j = j+(w[i]*(X.loc[[n], par[i-1]]))
    return j

def ztest(n):
    j = 0
    for i in range(0, 8):
        if i == 0:
            j = j + w[i]
        else:
            j = j+(w[i]*(T.loc[[n], par[i-1]]))
    return j

def htrain(x):
    return 1/(1+math.e**(-ztrain(x)))

def htest(x):
    return 1/(1+math.e**(-ztest(x)))

h_YList = [0]*891
def h_Y():
    for i in range(0, 891):
        h_YList[i] = float(htrain(i) - Y.loc[[i]])

#compute the partial derivative w.r.t wi
sumlist = [0]*8
def sumJp(x):
    s = 0
    if x == 0:
        s = sum(h_YList)
    else:
        for i in range(0, 891):
            print("------", h_YList[i]*float(X.loc[[i], par[x-1]]))
            s = s + h_YList[i]*float(X.loc[[i], par[x-1]])
    sumlist[x] = s/891
    print("---", s)
sumJp(2)
#update wi
def updateW():
    h_Y()
    for j in range(0, 8):
        sumJp(j)
    for i in range(0, 8):
        w[i] = w[i] - a*sumlist[i]

prediction = [1] * 418
def logistic():
    for i in range(0, 10000):
        updateW()
        print("-----", i, "-----")
        print(w)
    for i in range(0, 418):
        if float(htest(i)) > 0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0

#logistic()
submission = pd.DataFrame({
    "PassengerId": test_one["PassengerId"],
    "Survived": prediction
})
submission.to_csv("submission.csv", index=False)