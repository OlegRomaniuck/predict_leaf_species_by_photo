
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

data_train = pd.read_csv('leaf_features_color_train.csv')
data_train["type_leaf"] = data_train["type_leaf"] - 1

print(data_train.shape)

data_train.drop("Unnamed: 0", axis=1, inplace=True)
data_train.drop("name", axis=1, inplace=True)
print(data_train.columns)
X_train, y_train = data_train.drop('type_leaf', axis =1), data_train.type_leaf
data_validation = pd.read_csv('leaf_features_color_valid.csv')
print(data_validation.shape)
data_validation["type_leaf"] = data_validation["type_leaf"] - 1
data_validation.drop("Unnamed: 0", axis=1, inplace=True)
data_validation.drop("name", axis=1, inplace=True)


X_val, y_val = data_validation.drop('type_leaf', axis =1), data_validation.type_leaf
print(X_val.shape)
print(y_val.shape)


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_val)
eval_set=[(dvalid, 'eval')]

param = {'max_depth': '6', 'objective':'multi:softprob',
         'tree_method':'hist', 'num_class': 14,
         'eval_metric': 'mlogloss'}

bst = xgb.train(param, dtrain, num_boost_round=100,
                verbose_eval=10, evals=eval_set)

preds=bst.predict(dtest)
print(log_loss(y_val, preds))
pickle.dump(bst, open("xgboost_based_softprobe.dat", "wb"))
# [0]	eval-mlogloss:1.71868
# [10]	eval-mlogloss:0.714218
# [20]	eval-mlogloss:0.544165
# [30]	eval-mlogloss:0.469288
# [40]	eval-mlogloss:0.42663
# [50]	eval-mlogloss:0.400277
# [60]	eval-mlogloss:0.384194
# [70]	eval-mlogloss:0.371365
# [80]	eval-mlogloss:0.36189
# [90]	eval-mlogloss:0.354884
# [99]	eval-mlogloss:0.350305
# 0.3503043797563764

