from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

interval = 3000
max_iter = 3000
res_tr = []
res_t = []

x = [0.00001, 0.00003, 0.00006, 0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01]
# training stage
for tmp in x:
    for i in range(interval, max_iter + 1, interval):
    #     model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #             decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    #             max_iter=i, probability=False, shrinking=True,
    #             tol=0.001, verbose=False, random_state=random_seed)
        model = SVC(kernel='rbf', gamma=tmp, random_state=random_seed)
        model.fit(x_tr, y_tr)

    y_pre_tr = model.predict(x_tr)
    y_pre_t = model.predict(x_t)
    
    res_tr.append(round(f1_score(y_tr, y_pre_tr, average='micro'), 3))
    res_t.append(round(f1_score(y_t, y_pre_t, average='micro'), 3))

print('train: ', res_tr)
print('test: ', res_t)