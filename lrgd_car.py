import numpy as np
import pandas as pd
import matplotlib as plt

class LogisticRegressionGD(object):
    def __init__(self, eta = 0.05, n_iter = 100, random_state = 1, C = 100):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.C = C
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size = 1 + X.shape[1])
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            if self.C == 0:
                self.w_[1:] += self.eta * X.T.dot(errors)
            else:
                self.w_[1:] += self.eta * (X.T.dot(errors) + self.w_[1:] / self.C)
            self.w_[0] += self.eta * errors.sum()
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


X_train = pd.read_csv("dataset_files/car_X_train.csv")
y_train = pd.read_csv("dataset_files/car_y_train.csv")
X_test = pd.read_csv("dataset_files/car_X_test.csv")
y_test = pd.read_csv("dataset_files/car_y_test.csv")

dict_buying = dict()
dict_buying['low'] = 1
dict_buying['med'] = 2
dict_buying['high'] = 3
dict_buying['vhigh'] = 4
dict_maint = dict()
dict_maint['low'] = 1
dict_maint['med'] = 2
dict_maint['high'] = 3
dict_maint['vhigh'] = 4
dict_doors = dict()
dict_doors['2'] = 1
dict_doors['3'] = 2
dict_doors['4'] = 3
dict_doors['5more'] = 4
dict_persons = dict()
dict_persons['2'] = 1
dict_persons['4'] = 2
dict_persons['more'] = 3
dict_lug_boot = dict()
dict_lug_boot['small'] = 1
dict_lug_boot['med'] = 2
dict_lug_boot['big'] = 3
dict_safety = dict()
dict_safety['low'] = 1
dict_safety['med'] = 2
dict_safety['high'] = 3
dict_class = dict()
dict_class['unacc'] = 0
dict_class['acc'] = 1
dict_class['good'] = 2
dict_class['vgood'] = 3

for i in range(len(X_train)):
    X_train.iloc[i,0] = dict_buying[X_train.iloc[i,0]]
    X_train.iloc[i,1] = dict_maint[X_train.iloc[i,1]]
    X_train.iloc[i,2] = dict_doors[X_train.iloc[i,2]]
    X_train.iloc[i,3] = dict_persons[X_train.iloc[i,3]]
    X_train.iloc[i,4] = dict_lug_boot[X_train.iloc[i,4]]
    X_train.iloc[i,5] = dict_safety[X_train.iloc[i,5]]

for i in range(len(y_train)):
    y_train.iloc[i,0] = dict_class[y_train.iloc[i,0]]

for i in range(len(X_test)):
    X_test.iloc[i,0] = dict_buying[X_test.iloc[i,0]]
    X_test.iloc[i,1] = dict_maint[X_test.iloc[i,1]]
    X_test.iloc[i,2] = dict_doors[X_test.iloc[i,2]]
    X_test.iloc[i,3] = dict_persons[X_test.iloc[i,3]]
    X_test.iloc[i,4] = dict_lug_boot[X_test.iloc[i,4]]
    X_test.iloc[i,5] = dict_safety[X_test.iloc[i,5]]

for i in range(len(y_test)):
    y_test.iloc[i,0] = dict_class[y_test.iloc[i,0]]




test = pd.read_csv("test.csv")
for j in range(len(test)):
    for k in range(1, len(test.columns)):

        eta = float(test.iloc[j, 0])
        C = int(test.columns[k])
        n_iter = 100
        print(eta, C, n_iter)

        prob = []
        result = []
        for cl in range(4):
            ## Train model
            X_std = np.copy(X_train).astype(np.int)
            y_std = []
            for i in range(len(y_train)):
                y_std.append(y_train.iloc[i,0])
            y_std = np.copy(y_std)
            y_std = np.where(y_std == cl, 1, 0).astype(np.int)
            lrgd = LogisticRegressionGD(eta = eta, n_iter = n_iter, random_state = 1, C = C)
            lrgd.fit(X_std, y_std)

            ## Test model
            X_std = np.copy(X_test).astype(np.int)
            if len(prob) == 0:
                prob = lrgd.activation(lrgd.net_input(X_std))
                for i in range(len(prob)):
                    result.append(cl)
            else:
                prob_t = lrgd.activation(lrgd.net_input(X_std))
                for i in range(len(prob)):
                    if prob_t[i] >= prob[i]:
                        prob[i] = prob_t[i]
                        result[i] = cl

        cnt = 0
        for i in range(len(y_test)):
            if result[i] == y_test.iloc[i,0]:
                cnt += 1
        test.iloc[j,k] = cnt / len(y_test)

test.to_csv("lrgd_car_result.csv")
