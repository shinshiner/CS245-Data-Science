from sklearn.cross_validation import KFold
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt

##################### making dataset ######################

n = 500                 # number of instances
n_f = 30                # number of features
n_c = 3                 # number of classes
inf_f = int(0.6 * n_f)  # 60% real features
red_f = int(0.1 * n_f)  # 10% redundant features
rep_f = int(0.1 * n_f)  # 10% repeated features

X, Y = make_classification(n_samples=n, n_classes=n_c, flip_y=0.03,
                    n_features=n_f, n_informative=inf_f, n_redundant=red_f,
                    n_repeated=rep_f)
X_train, X_test, Y_train, Y_test = \
    train_test_split(X,Y, test_size=0.2)

##################### making dataset ######################

# directly train the model with training set & testing set
def exp_plain_train():
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    # pred = model.predict(X_test)
    # print(classification_report(Y_test, pred))
    score = model.score(X_test, Y_test)
    print('plain train score: ', score)

# test model with cross-validation
def exp_cv(folds=10):
    kfolds = KFold(n=Y.shape[0], n_folds=folds)
    model = DecisionTreeClassifier()
    scores = []
    for train, test in kfolds:
        model.fit(X[train], Y[train])
        # pred = model.predict(X[test])
        score = model.score(X[test], Y[test])
        scores.append(score)
        # print(score)
    print('avg score with cv folds %d: ' % folds, np.array(scores).mean())

# grid search
def exp_grid_search(folds=5):
    model = DecisionTreeClassifier()
    param_grid = {'criterion': ['gini', 'entropy']}
    grid = GridSearchCV(model, param_grid, cv=folds, scoring='accuracy')
    grid.fit(X, Y)
    print(grid.best_score_)

def bagging():
    bagging = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=100,       # number of models
        # bootstrap=True,
        # max_samples=1.0,        # Bootstrap sample size radio
        # bootstrap_features=True,
        # max_features=0.7,       # Bootstrap feature usage radio
    )
    bagging.fit(X_train, Y_train)

    pred = bagging.predict(X_test)
    print(classification_report(Y_test, pred))

def boosting():
    bagging = AdaBoostClassifier(
        DecisionTreeClassifier(),
        n_estimators=100,   # number of models
        algorithm='SAMME'   # Advanced-Boosting
    )
    bagging.fit(X_train, Y_train)

    pred = bagging.predict(X_train)
    print(classification_report(Y_train, pred))

    pred = bagging.predict(X_test)
    print(classification_report(Y_test, pred))

if __name__ == '__main__':
    # exp_plain_train()
    # exp_cv()
    exp_grid_search()
    # boosting()
    # bagging()