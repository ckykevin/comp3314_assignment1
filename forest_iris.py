import numpy as np
import pandas as pd
from random import randrange
from random import seed

def subsample(num, n):
    sample = []
    while len(sample) < n:
        index = randrange(num)
        sample.append(index)
    return sample

def test_split(index, value, sample, train_data, train_class):
    left, right = [], []
    for i in sample:
        if train_data.iloc[i, index] < value:
            left.append(i)
        else:
            right.append(i)
    return left, right

def gini_index(groups, classes, train_class):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [train_class.iloc[i,0] for i in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def get_split(sample, train_data, train_class, n_features):
    class_values = [0,1,2]
    features = []
    b_index, b_value, b_score, b_groups = 1000, 1000, 1000, None
    while len(features) < n_features:
        index = randrange(3)
        if index not in features:
            features.append(index)
    for index in features:
        for i in sample:
            groups = test_split(index, train_data.iloc[i,index], sample, train_data, train_class)
            gini = gini_index(groups, class_values, train_class)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, train_data.iloc[i, index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group, train_data, train_class):
    outcomes = [train_class.iloc[i,0] for i in group]
    return max(set(outcomes), key = outcomes.count)

def split(node, max_depth, n_features, depth, train_data, train_class):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right, train_data, train_class)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left, train_data, train_class), to_terminal(right, train_data, train_class)
        return
    if len(left) <= 1:
        node['left'] = to_terminal(left, train_data, train_class)
    else:
        node['left'] = get_split(left, train_data, train_class, n_features)
        split(node['left'], max_depth, n_features, depth+1, train_data, train_class)

    if len(right) <= 1:
        node['right'] = to_terminal(right, train_data, train_class)
    else:
        node['right'] = get_split(right, train_data, train_class, n_features)
        split(node['right'], max_depth, n_features, depth+1, train_data, train_class)

def build_tree(sample, train_data, train_class, max_depth, n_features):
    root = get_split(sample, train_data, train_class, n_features)
    split(root, max_depth, n_features, 1, train_data, train_class)
    return root

def predict(test_data, test_class, i, node):
    if test_data.iloc[i, node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(test_data, test_class, i, node['left'])
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(test_data, test_class, i, node['right'])
        else:
            return node['right']

def bagging_predict(test_data, test_class, i, trees):
    predictions = [predict(test_data, test_class, i, tree) for tree in trees]
    return max(set(predictions), key = predictions.count)

def random_forest(train_data, train_class, test_data, test_class, sample_size, n_features, n_trees, max_depth):
    trees = []
    for i in range(n_trees):
        sample = subsample(len(train_data), sample_size)
        tree = build_tree(sample, train_data, train_class, max_depth, n_features)
        trees.append(tree)
    prediction = []
    for i in range(len(test_data)):
        prediction.append(bagging_predict(test_data, test_class, i, trees))
    return(prediction)

seed(7)
X_train = pd.read_csv("dataset_files/iris_X_train.csv")
y_train = pd.read_csv("dataset_files/iris_y_train.csv")
X_test = pd.read_csv("dataset_files/iris_X_test.csv")
y_test = pd.read_csv("dataset_files/iris_y_test.csv")

test = pd.read_csv("test2.csv")
for i in range(len(test)):
    for j in range(1, len(test.columns)):
        print(i,j)
        ratio = int(test.iloc[i,0])
        n_trees = int(test.columns[j])
        prediction = random_forest(X_train, y_train, X_test, y_test, len(X_train) / ratio, 1, n_trees, 10)
        cnt = 0
        for k in range(len(y_test)):
            if prediction[k] == y_test.iloc[k,0]:
                cnt += 1
        test.iloc[i,j] = cnt / len(y_test)
test.to_csv("forest_iris_result.csv")
