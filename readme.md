# A 2-phase multi-class TrAdaboost
### NEW FEATURE
Add 2-phase sampling process when training the weak learner, firstly,
feed the weak classifier with weighted samples on target domain,
then get the predictions of the source data to select the subset
where the weighted residual is smaller than the threshold. In this
way, we could use all the weak learners and the influence of
dissimilar data on source domain can be mitigated.

### HOW TO USE
```python
clf = TrAdaBoostClassifier(n_source,
                 n_target,
                 base_estimator,
                 n_estimators,
                 learning_rate,
                 min_src_err)
clf.fit(train_data, train_target)
clf.predict_proba(new_data)
```
### REFERENCE
[1] *Dai W, Yang Q, Xue G R, et al. Boosting for transfer learning[C]// International Conference on Machine Learning. ACM, 2007:193-200.*
### CONCAT
pyk3350266@163.com