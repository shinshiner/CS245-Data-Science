from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
import numpy as np

def norm(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

random_seed = 666
np.random.seed(666)
# load data
X = np.load('adult_x.npy')
Y = np.load('adult_y.npy')

x_tr = np.load('adult_x_tr.npy')
x_t = np.load('adult_x_t.npy')
y_tr = np.load('adult_y_tr.npy')
y_t = np.load('adult_y_t.npy')

# interval = 3000
# max_iter = 3000
# res_tr = []
# res_t = []

# #x = [0.00001, 0.00003, 0.00006, 0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01]
# x = [0.03, 0.06, 0.1, 0.3, 0.6, 1.0]
# # training stage
# for tmp in x:
#     for i in range(interval, max_iter + 1, interval):
#     #     model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     #             decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#     #             max_iter=i, probability=False, shrinking=True,
#     #             tol=0.001, verbose=False, random_state=random_seed)
#         model = SVC(kernel='rbf', gamma=tmp, random_state=random_seed)
#         model.fit(x_tr, y_tr)

#     y_pre_tr = model.predict(x_tr)
#     y_pre_t = model.predict(x_t)
    
#     res_tr.append(round(f1_score(y_tr, y_pre_tr, average='micro'), 3))
#     res_t.append(round(f1_score(y_t, y_pre_t, average='micro'), 3))

# print('train: ', res_tr)
# print('test: ', res_t)


def boosting(n=5, cv=True):
    boosting = AdaBoostClassifier(
        DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=9, presort=True, random_state=random_seed),
        n_estimators=n,   # number of models
        algorithm='SAMME.R',  # Advanced-Boosting
        random_state=random_seed
    )
    if cv:      # using cross-validation
        scores_train = []
        scores_test = []
        kfolds = StratifiedKFold(Y, n_folds=10, random_state=random_seed)
        for train, test in kfolds:
            boosting.fit(X[train], Y[train])

            score = boosting.score(X[test], Y[test])
            scores_test.append(score)
            score = boosting.score(X[train], Y[train])
            scores_train.append(score)

        mean_test = np.array(scores_test).mean()
        mean_train = np.array(scores_train).mean()
        print('avg score with cv folds 10 in testing set: ', mean_test)
        print('avg score with cv folds 10 in training set: ', mean_train)
        
        return mean_train, mean_test
    else:       # without cross-validation
        boosting.fit(X_train, Y_train)

        pred = boosting.predict(X_train)
        print(classification_report(Y_train, pred))

        pred = boosting.predict(X_test)
        print(classification_report(Y_test, pred))

    # plot the relation between weights and error
#     plt.figure()
#     plt.xlabel(u'子模型权重')
#     plt.ylabel(u'错误率')
#     plt.plot(boosting.estimator_weights_, boosting.estimator_errors_)
#     plt.savefig('report/img/boosting-weight-error-%d' % len(boosting.estimator_weights_))
#     plt.show()


res_tr = []
res_t = []

for i in range(10, 101, 10):
    tr, t = boosting(n=i)
    res_tr.append(round(tr, 3))
    res_t.append(round(t, 3))
    
print('train results: ', res_tr)
print('test results:: ', res_t)