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


X_train = pd.read_csv("dataset_files/iris_X_train.csv")
y_train = pd.read_csv("dataset_files/iris_y_train.csv")
X_test = pd.read_csv("dataset_files/iris_X_test.csv")
y_test = pd.read_csv("dataset_files/iris_y_test.csv")


test = pd.read_csv("test.csv")
for j in range(len(test)):
    for k in range(1, len(test.columns)):

        eta = float(test.iloc[j, 0])
        C = int(test.columns[k])
        n_iter = 100
        print(eta, C, n_iter)

        prob = []
        result = []
        for cl in range(3):
            ## Train model
            X_std = np.copy(X_train)
            y_std = []
            for i in range(len(y_train)):
                y_std.append(y_train.iloc[i,0])
            y_std = np.copy(y_std)
            y_std = np.where(y_std == cl, 1, 0)
            lrgd = LogisticRegressionGD(eta = eta, n_iter = n_iter, random_state = 1, C = C)
            lrgd.fit(X_std, y_std)

            ## Test model
            X_std = np.copy(X_test)
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

test.to_csv("lrgd_iris_result.csv")
