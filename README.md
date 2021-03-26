# comp3314_assignment1
HKU COMP3314 Machine Learning

1. Multiclass logistic regression

Simply run lrgd_iris.py and lrgd_car.py. For n_iter = 100, the result is contained in lrgd_iris_result.csv and lrgd_car_result.csv. For n_iter = 500, the result is contained in lrgd_iris_result_2.csv and lrgd_car_result_2.csv. They will extract data from test.csv which contains different learning rate and constant C.

2. Random forest

Simply run forest_car.py and forest_iris.py. In the main part of those programs, (forest_car.py from line 196, forest_iris.py from line 140), we firstly implement random forest using fixed settings. Next, we use a loop to see how different parameters influence the final result. We can use the parameter as the iterable variable and hide its fixed settings to see their performance.

