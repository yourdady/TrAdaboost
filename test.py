''' 
@project TrAdaboost
@author Peng
@file test.py
@time 2018-08-16
'''
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer,load_digits,load_wine
from TrAdaboost import TrAdaBoostClassifier
def main():
    num_train = 400
    clf = TrAdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                               n_source=200, n_target=200, n_estimators=50, learning_rate=0.05, min_src_err=0.001)

    bc = load_breast_cancer()
    print(len(bc.data))
    data = bc.data
    target = bc.target
    train_data = data[:num_train]
    train_target = target[:num_train]
    test_data = data[num_train:]
    test_target = target[num_train:]

    clf.fit(train_data, train_target)
    acc = clf.score_accuracy(test_data, test_target,begin=3)
    clf.draw_plot()
    print('acc',acc)
if __name__ == '__main__':
    main()