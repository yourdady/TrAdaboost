# A 2-phase multi-class TrAdaboost
### NEW FEATURE
A 2-phase sampling process is introduced in this project when training the weak learners.
firstly, feed the weak learners with weighted samples on target domain,
then get the predictions of the source data to select the subset
in which the sample's weighted residual is smaller than the threshold. In this
way, we could use all the weak learners and the influence of
dissimilar data on source domain can be mitigated.

### HOW TO USE
```python
    clf = TrAdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                               n_source=200, n_target=200, n_estimators=100,
                               learning_rate=0.03, min_src_err=0.0005)

    clf.fit(train_data, train_target)
    clf.predict_proba(new_data)
```
### REFERENCE
[1] *Dai W, Yang Q, Xue G R, et al. Boosting for transfer learning[C]// International Conference on Machine Learning. ACM, 2007:193-200.*
### CONCAT
pyk3350266@163.com