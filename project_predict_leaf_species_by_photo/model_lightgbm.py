import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc


data_train = pd.read_csv('leaf_features_color_train.csv')
data_train["type_leaf"] = data_train["type_leaf"] - 1

print(data_train.shape)

data_train.drop("Unnamed: 0", axis=1, inplace=True)
data_train.drop("name", axis=1, inplace=True)
# data_train.to_csv("data_train_color_test_normilized.csv", index=False)
print(data_train.columns)

X_train, y_train = data_train.drop('type_leaf', axis =1), data_train.type_leaf


data_validation = pd.read_csv('leaf_features_color_valid.csv')
print(data_validation.shape)
data_validation["type_leaf"] = data_validation["type_leaf"] - 1
data_validation.drop("Unnamed: 0", axis=1, inplace=True)
data_validation.drop("name", axis=1, inplace=True)
# data_validation.to_csv("data_valid_color_test_normilized.csv", index=False)

X_val, y_val = data_validation.drop('type_leaf', axis =1), data_validation.type_leaf
print(X_val.shape)
print(y_val.shape)


# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
#
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#
# parameters = {
#     # default
#     "objective": "binary",
#     "learning_rate": 0.1,
#     "num_threads": 10,
#     "metric": "auc",
#     "seed": 42,
#
#     # regularization
#     "colsample_bytree": 0.8,
#     "subsample": 0.8,
#     "subsample_freq": 1,
#     "min_data_in_leaf": 15
# }
# 'multi_error'
# specify your configurations as a dict
# params = {
#           "objective" : "multiclass",
#           'metric': {'multi_logloss'},
#           "num_class" : 14,
#           "num_leaves" : 80,
#           "max_depth": -1,
#           "learning_rate" : 0.01,
#           "bagging_fraction" : 0.9,  # subsample
#           "feature_fraction" : 0.9,  # colsample_bytree
#           "bagging_freq" : 5,        # subsample_freq
#           "bagging_seed" : 2018,
#           "verbosity" : -1 ,
#           # "early_stopping_rounds":100,
#           "shuffle": True,
#           "stratified": True}
# #
# print('Starting training...')
# # train
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=50,
#                 valid_sets=lgb_eval,
#                 early_stopping_rounds=30)
# #
# print('Saving model...')
# # save model to file
# gbm.save_model('model_ligtgbm.txt')
gbm = lgb.Booster(model_file='model_ligtgbm.txt')
#
# print('Starting predicting...')
# # predict
y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
for index, instance in data_validation.iterrows():
    prediction = np.argmax(y_pred[index])

    print("Actual {} Prediction {}".format(y_val[index], prediction))

predictions_classes = []
for i in y_pred:
    predictions_classes.append(np.argmax(i))
predictions_classes = np.array(predictions_classes)
accuracy = accuracy_score(predictions_classes, y_val)*100
print("ACC ==> {}".format(accuracy))
actuals_onehot = pd.get_dummies(y_val).values
false_positive_rate, recall, thresholds = roc_curve(actuals_onehot[0], np.round(y_pred)[0])
roc_auc = auc(false_positive_rate, recall)
print("AUC score ", roc_auc)
# ACC ==> 59.76429426387993
# AUC score  0.5
# # eval
# print('The rmse of prediction is:', mean_squared_error(y_val, y_pred) ** 0.5)
