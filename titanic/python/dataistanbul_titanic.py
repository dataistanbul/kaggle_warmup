# coding: utf-8

# pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get titanic & test csv files as a DataFrame
from scipy.stats._continuous_distns import foldcauchy_gen

train_df = pd.read_csv("../../input/train.csv", dtype={"Age": np.float64}, )
# print train_df.info()
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Name           891 non-null object
# Sex            891 non-null object
# Age            714 non-null float64       missing 177
# SibSp          891 non-null int64
# Parch          891 non-null int64
# Ticket         891 non-null object
# Fare           891 non-null float64
# Cabin          204 non-null object        missing 687
# Embarked       889 non-null object        missing 2

test_df = pd.read_csv("../../input/test.csv", dtype={"Age": np.float64}, )
# print test_df.info()
# PassengerId    418 non-null int64
# Pclass         418 non-null int64
# Name           418 non-null object
# Sex            418 non-null object
# Age            332 non-null float64       missing 86
# SibSp          418 non-null int64
# Parch          418 non-null int64
# Ticket         418 non-null object
# Fare           417 non-null float64       missing 1
# Cabin          91 non-null object         missing 326
# Embarked       418 non-null object

# Missing Totals: Age (263), Cabin (1013), Fare (1), Embarked (2)

titanic_df = train_df.copy().drop(["Survived"], axis=1)
titanic_df = titanic_df.append(test_df, ignore_index=True)
# print titanic_df.info()
# PassengerId    1309 non-null int64
# Pclass         1309 non-null int64
# Name           1309 non-null object
# Sex            1309 non-null object
# Age            1046 non-null float64      missing 263
# SibSp          1309 non-null int64
# Parch          1309 non-null int64
# Ticket         1309 non-null object
# Fare           1308 non-null float64      missing 1
# Cabin          295 non-null object        missing 1013
# Embarked       1307 non-null object       missing 2


def get_titles():
    global titanic_df

    # we extract the title from each name
    titanic_df['Title'] = titanic_df['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }

    titanic_df['Title'] = titanic_df.Title.map(Title_Dictionary)

get_titles()

def process_age():
    global titanic_df

    # a function that fills the missing values of the Age variable

    def fillAges(row):
        if row['Sex'] == 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex'] == 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex'] == 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex'] == 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex'] == 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex'] == 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26

    titanic_df.Age = titanic_df.apply(lambda r: fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)

process_age()


def process_names():
    global titanic_df
    # we clean the Name variable
    titanic_df.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(titanic_df['Title'], prefix='Title')
    titanic_df = pd.concat([titanic_df, titles_dummies], axis=1)

    # removing the title variable
    titanic_df.drop('Title', axis=1, inplace=True)

process_names()

# Convert categorized to one-hot feature set
sex_dummies = pd.get_dummies(titanic_df['Sex'], prefix='Sex', dummy_na=True)
embarked_dummies = pd.get_dummies(titanic_df['Embarked'], prefix='Embarked', dummy_na=True)

# merge the two one-hot feature sets
titanic_df = pd.concat([titanic_df, sex_dummies], axis=1)
titanic_df = pd.concat([titanic_df, embarked_dummies], axis=1)

# Add relatives as a new feature
titanic_df["Relatives"] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

# Fill Fare
if len(titanic_df.Fare[titanic_df.Fare.isnull()]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = titanic_df[ titanic_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        titanic_df.loc[ (titanic_df.Fare.isnull()) & (titanic_df.Pclass == f+1 ), 'Fare'] = median_fare[f]


# Drop unused features
titanic_df = titanic_df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
titanic_df = titanic_df.drop(["Sex", "Embarked"], axis=1)

X_train = titanic_df[0:train_df.shape[0]]
Y_train = train_df["Survived"]
X_test = titanic_df[train_df.shape[0]:]

# --------------------------
from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier() # 0.69856
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=4) # 0.76077
clf.fit(X_train,Y_train)
print "DTC Prediction score:", clf.score(X_train, Y_train)
predictions_dtc = clf.predict(X_test)
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'], 'Survived': predictions_dtc })
submission.to_csv("submission-kince-dtc.csv", index=False)
# --------------------------
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=240, learning_rate=0.1, max_depth=3) # 0.76555
clf.fit(X_train,Y_train)
print "GBC Prediction score:", clf.score(X_train, Y_train)
predictions_gbc = clf.predict(X_test)
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'], 'Survived': predictions_gbc })
submission.to_csv("submission-kince-gbc.csv", index=False)
# --------------------------
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', leaf_size=50)
clf.fit(X_train,Y_train)
print "KNC2 Prediction score:", clf.score(X_train, Y_train)
predictions_knc2 = clf.predict(X_test)
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'], 'Survived': predictions_knc2 })
submission.to_csv("submission-kince-knc2.csv", index=False)
# --------------------------

exit(0)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(X_train, Y_train)
features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
#print features.sort(['importance'],ascending=False)

# model = SelectFromModel(clf, prefit=True)
# train_new = model.transform(X_train)
# print train_new.shape
#
# test_new = model.transform(X_test)
# print test_new.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

# forest = RandomForestClassifier(max_features='sqrt')
# parameter_grid = {
#                  'max_depth' : [3,4,5,6,7,8],
#                  'n_estimators': [190,200,210,240,250],
#                  'criterion': ['gini','entropy']
#                  }
#
# cross_validation = StratifiedKFold(Y_train, n_folds=5)
# grid_search = GridSearchCV(forest,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)
#
# grid_search.fit(X_train, Y_train)
#
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

# forest = DecisionTreeClassifier(max_features='sqrt')
# print forest.get_params().keys()
#
# parameter_grid = {
#                  'max_depth' : [3,4,5,6,7,8],
#                  'min_samples_leaf': [1,2,3,4,5],
#                  'criterion': ['gini','entropy']
#                  }
#
# cross_validation = StratifiedKFold(Y_train, n_folds=5)
# grid_search = GridSearchCV(forest,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)
#
# grid_search.fit(X_train, Y_train)
#
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))
#
# forest = GradientBoostingClassifier(max_features='sqrt')
# print forest.get_params().keys()
#
# parameter_grid = {
#                  'max_depth' : [3,4,5,6,7,8],
#                  'n_estimators': [190,200,210,240,250],
#                  'learning_rate': [0.1, 0.01, 0.001, 0.0001]
#                  }
#
# cross_validation = StratifiedKFold(Y_train, n_folds=5)
# grid_search = GridSearchCV(forest,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)
#
# grid_search.fit(X_train, Y_train)
#
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

forest = KNeighborsClassifier()
print forest.get_params().keys()

parameter_grid = {
                 'n_neighbors' : [2,3,4,5,6,7,8],
                 'algorithm': ['ball_tree', 'kd_tree', 'auto', 'brute'],
                 'leaf_size': [10, 20, 30, 40, 50]
                 }

cross_validation = StratifiedKFold(Y_train, n_folds=5)
grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(X_train, Y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
