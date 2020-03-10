import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold

data_train = pd.read_csv(
    'data_train_color_test_normilized.csv')
X_train, y_train = data_train.drop('type_leaf', axis=1), data_train.type_leaf
data_validation = pd.read_csv(
    'data_valid_color_test_normilized.csv')
X_val, y_val = data_validation.drop('type_leaf', axis=1), data_validation.type_leaf

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# eval_metric merror map auc
xgb_clf_1 = xgb.XGBClassifier(verbosity=1, eval_metric=["merror", "mlogloss", "auc"], objective="multi:softmax",
                              random_state=42)
# # Create parameter grid
parameters = {"max_depth": [2, 5, 7, 14, 20, 30, 60], "eta": [1, 2, 4], 'n_estimators': [50, 100, 200],
              "learning_rate": [0.1, 0.5, 1.0, 3.0, 7.0], "subsample": [1, 3, 7], "colsample_bytree": [1, 3, 7],
              "num_class": [14]}

# # Create RandomizedSearchCV Object
xgb_rscv = GridSearchCV(estimator=xgb_clf_1, param_grid=parameters, scoring="f1_micro", cv=skf, verbose=1, n_jobs=4)
xgb_rscv.fit(X_train, y_train)

print(xgb_rscv.best_estimator_)
print(xgb_rscv.best_score_)
print(xgb_rscv.best_params_)

# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, eta=1,
#               eval_metric=['merror', 'mlogloss', 'auc'], gamma=0,
#               learning_rate=0.5, max_delta_step=0, max_depth=5,
#               min_child_weight=1, missing=None, n_estimators=200, n_jobs=1,
#               nthread=None, num_class=14, objective='multi:softprob',
#               random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#               seed=None, silent=None, subsample=1, verbosity=1)
# 0.8873304479467492
# {'colsample_bytree': 1, 'eta': 1, 'learning_rate': 0.5, 'max_depth': 5, 'n_estimators': 200, 'num_class': 14, 'subsample': 1}
