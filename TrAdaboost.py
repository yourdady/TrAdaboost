''' 
@project TrAdaboost
@author Peng
@file TrAdaboost.py
@time 2018-06-13
'''
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from utils import softmax



class TrAdaBoostClassifier(object):
    '''
    A two-phase multi-class TrAdaboost providing API in sklearn style, which performs better 
    than original TrAdaboost cause more than N/2 estimators can be utilzied. 
    
    '''
    def __init__(self,
                 n_source,
                 n_target,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1,
                 min_src_err=1
                 ):
        """
        
        :param n_source: int, 
        :param n_target: int, 
        :param base_estimator:  
        :param n_estimators: int,
        :param learning_rate: float,
        :param min_src_err: float,
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.n_source = n_source
        self.n_target = n_target
        self.estimator_err_source = 1/(1 + math.sqrt(2 * math.log(float(n_source / self.n_estimators))))
        self.source_weight_ratio = []
        self.source_weight = []
        self.target_weight = []
        self.iteration_accuracy = []
        self.one_hot = False
        self.min_src_err = min_src_err

    def fit(self, X, y, sample_weight = None, early_stop_err = 0):
        '''
            Parameters
            ----------
            X : 2-d array-like of shape=[n_source+n_target, n_features].
            
            y : int, number of target samples.
            
            sample_weight : 1-d array like of shape=[n_source+n_target], 1/(n_source+n_target) 
            by default.
            
            one_hot : bool, if y is one-hot coding.
            
            early_stop_err : float, threshold for early stopping.
            
        '''
        if sample_weight is None:
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = sample_weight/sample_weight.sum(dtype=np.float64)
        if X.shape[0] != self.n_target + self.n_source:
            raise ValueError("n_target + n_source != samplenum")
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        for iboost in range(self.n_estimators):
            print("训练第",iboost,"个分类器")
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                self.n_source,
                self.n_target,
                )
            sample_weight /= np.sum(sample_weight)
            if estimator_weight!=0:
                self.estimator_weights_[iboost] = estimator_weight
            else:
                self.n_estimators = iboost
                print("提前结束，迭代次数: ",iboost)
                break
            self.estimator_errors_[iboost] = estimator_error

            if estimator_error < early_stop_err:
                self.n_estimators = iboost
                print("提前结束，迭代次数: ",iboost)
                break
            sample_weight_sum = np.sum(sample_weight)
            source_weight = np.sum(sample_weight[:self.n_source])/sample_weight_sum
            self.source_weight_ratio.append(source_weight)
            self.source_weight.append(np.sum(sample_weight[:self.n_source]))
            self.target_weight.append(np.sum(sample_weight[-self.n_target:]))
        return self

    #对独热编码进行转换
    def transform_label_from_one_hot(selfs, label):
        return np.argmax(label,axis=1)


    def _boost(self, iboost, X, y, sample_weight, n_source, n_target):
        """
        
        :param iboost: 
        :param X: 
        :param y: 
        :param sample_weight: 
        :param n_source: 
        :param n_target: 
        :return: 
        """
        estimator = self._make_estimator()
        print("正在fit第",iboost,"个分类器")

        estimator.fit(X[-n_target:], y[-n_target:], sample_weight=sample_weight[-n_target:])
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        source_proba = np.array(estimator.predict_proba(X[:n_source]))
        source_pred = np.array([source_proba[i][y[i]] for i in range(source_proba.shape[0])])
        if self.one_hot == True:
            e = np.abs(source_pred - self.transform_label_from_one_hot(y[:n_source]))
        else:
            e = np.abs(source_pred - y[:n_source])

        index = np.where(sample_weight[:n_source] * e < self.min_src_err)
        print('number of trainable samples of source', len(index[0]))
        X_train = np.concatenate((X[-n_target:], X[:n_source][index]), axis=0)
        y_train = np.concatenate((y[-n_target:], y[:n_source][index]), axis=0)
        sample_weight_train = np.concatenate((sample_weight[-n_target:], sample_weight[:n_source][index]),
                                             axis=0)

        if self.one_hot == False:
            estimator.fit(X_train, y_train, sample_weight = sample_weight_train)
        else:
            estimator.fit(X_train, self.transform_label_from_one_hot(y_train), sample_weight = sample_weight_train)
        print("fit完毕")
        y_predict = estimator.predict(X)


        if self.one_hot == True:
            e = np.abs(y_predict - self.transform_label_from_one_hot(y))/self.n_classes_
        else:
            e = np.abs(y_predict - y)/self.n_classes_

        e_target = e[-n_target:]
        e_source = e[:n_source]
        sample_weight_target = sample_weight[-n_target:]
        # sample_weight_source = sample_weight[:n_source]
        err = np.average(e_target, weights=sample_weight_target)
        estimator_weight_target = err / ((1 - err)*(self.n_classes_-1))
        estimator_weight_source = self.estimator_err_source/(self.n_classes_-1)
        coefficient_source = np.array([estimator_weight_source**(i*self.learning_rate) for i in e_source])
        coefficient_target = np.array([estimator_weight_target**(-i*self.learning_rate) for i in e_target])

        if not iboost == self.n_estimators - 1:
            sample_weight[:n_source] = sample_weight[:n_source] * coefficient_source
            sample_weight[-n_target:] = sample_weight[-n_target:] * coefficient_target
        return sample_weight,estimator_weight_target,np.sum(e_target)


    def predict_proba(self,X, one_hot=False, begin = 0):
        """
        
        :param X: 
        :param one_hot: 
        :param begin: 
        :return: 
        """
        proba = np.ones(shape=[len(X), self.n_classes_])
        for i in range(begin , self.n_estimators):
            # proba *= self.estimator_weights_[i] ** \
            #          (-np.swapaxes(np.array(self.estimators_[i].predict_proba(X))[:, :, 1], 1, 0))
            proba[:,] *= self.estimator_weights_[i] ** -self.estimators_[i].predict_proba(X)
        # normaliz = np.log(reduce(lambda x, y: x * y, self.estimator_weights_))
        # proba = np.log(proba) / normaliz
        proba = softmax(np.log(proba))
        return proba

    def predict(self,X, one_hot=False, begin=0):
        proba = self.predict_proba(X,one_hot=one_hot, begin=begin)
        result = np.argmax(proba,axis=1)
        return  result


    def score_accuracy(self,X,y,begin=0):
        if self.one_hot == False:
            tmp = self.predict(X,begin=begin) == y
        else: tmp = self.predict(X,begin=begin) == np.argmax(y, axis=1)
        accuracy = np.sum(tmp)/len(X)
        return accuracy


    def _make_estimator(self):
        estimator = copy.deepcopy(self.base_estimator)
        self.estimators_.append(estimator)
        return estimator





if __name__ == '__main__':
    pass
