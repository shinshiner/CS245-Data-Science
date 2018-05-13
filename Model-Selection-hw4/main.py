from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from pydotplus import graph_from_dot_data

##################### making dataset ######################

n = 500                 # number of instances
n_f = 30                # number of features
n_c = 3                 # number of classes
inf_f = int(0.6 * n_f)  # 60% real features
red_f = int(0.1 * n_f)  # 10% redundant features
rep_f = int(0.1 * n_f)  # 10% repeated features
random_seed = 1         # random seed for the experiments

X, Y = make_classification(n_samples=n, n_classes=n_c, flip_y=0.03,
                    n_features=n_f, n_informative=inf_f, n_redundant=red_f,
                    n_repeated=rep_f, random_state=random_seed)
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.8, random_state=random_seed)

##################### making dataset ######################

# directly train the model with training set & testing set
def exp_plain_train():
    model = DecisionTreeClassifier(random_state=random_seed)
    model.fit(X_train, Y_train)

    pred = model.predict(X_test)
    print(classification_report(Y_test, pred))

    score = model.score(X_test, Y_test)
    print('plain train score in testing set: ', score)
    score = model.score(X_train, Y_train)
    print('plain train score in training set: ', score)

# test model with cross-validation
def exp_cv(folds=10):
    kfolds = StratifiedKFold(Y, n_folds=folds, random_state=random_seed)
    model = DecisionTreeClassifier(random_state=random_seed)
    scores_train = []
    scores_test = []
    for train, test in kfolds:
        model.fit(X[train], Y[train])
        # pred = model.predict(X[test])

        score = model.score(X[test], Y[test])
        scores_test.append(score)
        score = model.score(X[train], Y[train])
        scores_train.append(score)

    mean_test = np.array(scores_test).mean()
    mean_train = np.array(scores_train).mean()
    print('avg score with cv folds %d in testing set: '
          % folds, mean_test)
    print('avg score with cv folds %d in training set: '
          % folds, mean_train)

    return mean_test, mean_train

# grid search
def exp_grid_search(folds=10):
    model = DecisionTreeClassifier(random_state=random_seed)
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_features': ['sqrt', 'log2', None],
                  'max_depth': list(range(3, 15)),
                  'presort': [True, False],
                  'splitter': ['best', 'random']}
    grid = GridSearchCV(model, param_grid, cv=folds, scoring='f1_weighted')
    grid.fit(X, Y)

    print(grid.best_params_)
    print(grid.best_score_)

    export_graphviz(grid.best_estimator_, filled=True, out_file='report/img/gs.dot')

def bagging():
    bagging = BaggingClassifier(
        DecisionTreeClassifier(random_state=random_seed),
        n_estimators=100,       # number of models
        random_state=random_seed
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
        DecisionTreeClassifier(random_state=random_seed),
        n_estimators=100,   # number of models
        algorithm='SAMME',   # Advanced-Boosting
        random_state=random_seed
    )
    bagging.fit(X_train, Y_train)

    pred = bagging.predict(X_train)
    print(classification_report(Y_train, pred))

    pred = bagging.predict(X_test)
    print(classification_report(Y_test, pred))

def plot_cv():
    mean_tests = []
    mean_trains = []
    folds_sum = 11
    for i in range(2, folds_sum):
        m1, m2 = exp_cv(i)
        mean_tests.append(m1)
        mean_trains.append(m2)

    # start to plot
    x = np.arange(2, folds_sum)
    total_width, n = 0.8, 2
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, mean_tests, width=width,
            facecolor='#9999ff', edgecolor='white', label=u'测试集')
    plt.bar(x + width, mean_trains, width=width,
            facecolor='#ffa07a', edgecolor='white', label=u'训练集')
    for x, y1, y2 in zip(x, mean_tests, mean_trains):
        plt.text(x - 0.05, y1 + 0.01, '%.2f' % y1, ha='center', va='bottom')
        plt.text(x+width - 0.05, y2 + 0.01, '%.2f' % y2, ha='center', va='bottom')

    plt.xlabel(u'折数')
    plt.ylabel(u'平均 f1-score')
    plt.ylim((0, 1.3))
    plt.legend()
    plt.savefig('report/img/cv_bar')
    plt.show()

if __name__ == '__main__':
    # exp_plain_train()
    # exp_cv()
    # plot_cv()
    exp_grid_search()
    # boosting()
    # bagging()