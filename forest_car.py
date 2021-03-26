import numpy as np
import pandas as pd
from random import randrange
from random import seed

def sub(num, n):
    sample = []
    while len(sample) < n:
        index = randrange(num)
        sample.append(index)
    return sample

def test_split(index, value, sample):
    left, right = [], []
    for i in sample:
        if train_data[i][index] < value:
            left.append(i)
        else:
            right.append(i)
    return left, right

def gini_value(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [train_class[i][0] for i in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def get(sample, n_features):
    class_values = [0,1,2,3]
    features = []
    b_score= 1000
    while len(features) < n_features:
        index = randrange(6)
        if index not in features:
            features.append(index)
    for index in features:
        for i in sample:
            groups = test_split(index, train_data[i][index], sample)
            gini = gini_value(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, train_data[i][index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def output(group):
    outcomes = [train_class[i][0] for i in group]
    return max(set(outcomes), key = outcomes.count)

def split(node, max_depth, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['l'] = node['r'] = output(left + right)
        return
    if depth >= max_depth:
        node['l'], node['r'] = output(left), output(right)
        return
    if len(left) <= 1:
        node['l'] = output(left)
    else:
        node['l'] = get(left, n_features)
        split(node['l'], max_depth, n_features, depth+1)

    if len(right) <= 1:
        node['r'] = output(right)
    else:
        node['r'] = get(right, n_features)
        split(node['r'], max_depth, n_features, depth+1)

def build(sample, max_depth, n_features):
    root = get(sample, n_features)
    split(root, max_depth, n_features, 1)
    return root

def predict(i, node):
    if test_data[i][node['index']] < node['value']:
        if isinstance(node['l'], dict):
            return predict(i, node['l'])
        else:
            return node['l']
    else:
        if isinstance(node['r'], dict):
            return predict(i, node['r'])
        else:
            return node['r']

def random_forest(sample_size, n_features, n_trees, max_depth):
    trees = []
    for i in range(n_trees):
        sample = sub(len(train_data), sample_size)
        tree = build(sample, max_depth, n_features)
        trees.append(tree)
    prediction = []
    for i in range(len(test_data)):
        t = [predict(i, tree) for tree in trees]
        prediction.append(max(set(t), key = t.count))
    return(prediction)

seed(7)
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

train_data = []
train_class = []
test_data = []
test_class = []

for i in range(len(X_train)):
    data = []
    for j in range(len(X_train.columns)):
        data.append(X_train.iloc[i,j])
    train_data.append(data)

for i in range(len(y_train)):
    data = []
    for j in range(len(y_train.columns)):
        data.append(y_train.iloc[i,j])
    train_class.append(data)

for i in range(len(X_test)):
    data = []
    for j in range(len(X_test.columns)):
        data.append(X_test.iloc[i,j])
    test_data.append(data)

for i in range(len(y_test)):
    data = []
    for j in range(len(y_test.columns)):
        data.append(y_test.iloc[i,j])
    test_class.append(data)

ratio = 6
n_trees = 75
prediction = random_forest(len(X_train) / ratio, 4, n_trees, 8)
cnt = 0
for k in range(len(y_test)):
    if prediction[k] == y_test.iloc[k,0]:
        cnt += 1
print(cnt / len(y_test))

for n_features in range(1,7):
    #n_features = 5
    max_depth = 7
    n_trees = 75
    ratio = 5
    prediction = random_forest(len(X_train) / ratio, n_features, n_trees, max_depth)
    cnt = 0
    for k in range(len(y_test)):
        if prediction[k] == y_test.iloc[k,0]:
            cnt += 1
    print('n_features is', n_features, 'accuracy is', cnt / len(y_test))
