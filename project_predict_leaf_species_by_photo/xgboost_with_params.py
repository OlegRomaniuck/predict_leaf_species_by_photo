import pickle
import pandas as pd
import xgboost as xgb
import itertools
import numpy as np

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from matplotlib import pyplot as plt

data_train = pd.read_csv(
    'data_train_color_test_normilized.csv')
X_train, y_train = data_train.drop('type_leaf', axis=1), data_train.type_leaf

data_validation = pd.read_csv(
    'data_valid_color_test_normilized.csv')
X_val, y_val = data_validation.drop('type_leaf', axis=1), data_validation.type_leaf

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
eval_set = [(dvalid, 'eval')]
param = {'colsample_bytree': 1, "objective": "multi:softmax", 'eval_metric': ["merror", "mlogloss"],
         'subsample': 1, 'num_class': 14, 'learning_rate': 0.5, 'max_depth': 5, 'eta': 1}
bst = xgb.cv(param, dtrain, num_boost_round=100, verbose_eval=10, early_stopping_rounds=10, folds=skf)

# [0]	train-merror:0.318909+0.00453696	train-mlogloss:1.36863+0.00529824	test-merror:0.338612+0.00429536	test-mlogloss:1.41342+0.00427937
# [10]	train-merror:0.15196+0.00170141	train-mlogloss:0.502406+0.00215705	test-merror:0.203209+0.00251367	test-mlogloss:0.640694+0.00304462
# [20]	train-merror:0.0939416+0.00141061	train-mlogloss:0.323385+0.00264135	test-merror:0.16847+0.00111458	test-mlogloss:0.50986+0.00720991
# [30]	train-merror:0.0604914+0.00104606	train-mlogloss:0.232396+0.00200516	test-merror:0.150218+0.000969038	test-mlogloss:0.451939+0.00998587
# [40]	train-merror:0.040109+0.000751576	train-mlogloss:0.175478+0.00134547	test-merror:0.142184+0.00126873	test-mlogloss:0.42183+0.00985157
# [50]	train-merror:0.0259382+0.000905298	train-mlogloss:0.136093+0.000918363	test-merror:0.135241+0.00171707	test-mlogloss:0.401633+0.0100054
# [60]	train-merror:0.0166556+0.000629535	train-mlogloss:0.107314+0.00105793	test-merror:0.130782+0.00223845	test-mlogloss:0.388642+0.010231
# [70]	train-merror:0.0100316+0.00055596	train-mlogloss:0.0855456+0.00115263	test-merror:0.127345+0.00261885	test-mlogloss:0.379144+0.0113584
# [80]	train-merror:0.0058576+0.000299253	train-mlogloss:0.0690234+0.000834972	test-merror:0.124536+0.00127317	test-mlogloss:0.373211+0.0117493
# [90]	train-merror:0.0031522+0.000462686	train-mlogloss:0.0559814+0.000969154	test-merror:0.122353+0.00147614	test-mlogloss:0.369106+0.0112767
# [99]	train-merror:0.001614+0.000245545	train-mlogloss:0.0467456+0.000683151	test-merror:0.120634+0.00181376	test-mlogloss:0.366623+0.0120542
bst = xgb.XGBClassifier(max_depth=5, learning_rate=0.5, n_estimators=100, objective="multi:softmax", colsample_bytree=1,
                        subsample=1, random_state=42)
model = bst.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric="mlogloss")
predicts_on_valid = model.predict(X_val)
print("Accuracy:", metrics.accuracy_score(y_val, predicts_on_valid))
print("Training F1 Micro Average: {}".format(f1_score(y_val, predicts_on_valid, average="micro")))
# Accuracy: 0.8744130374735292
# Training F1 Micro Average: 0.8744130374735292
pickle.dump(model, open("xgboost_better.dat", "wb"))
results = cross_val_score(model, X_val, y_val, cv=skf)
# evaluate accuracy on validation data set
print("CV accuracy score: {:.2f}%".format(results.mean() * 100))
print(classification_report(y_val, predicts_on_valid))
print(confusion_matrix(y_val, predicts_on_valid))
accuracy = accuracy_score(y_val, predicts_on_valid)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# [0]	validation_0-mlogloss:1.37289	validation_1-mlogloss:1.42727
# Multiple eval metrics have been passed: 'validation_1-mlogloss' will be used for early stopping.
#
# Will train until validation_1-mlogloss hasn't improved in 10 rounds.
# [1]	validation_0-mlogloss:1.12196	validation_1-mlogloss:1.19044
# [2]	validation_0-mlogloss:0.967323	validation_1-mlogloss:1.04877
# [3]	validation_0-mlogloss:0.853026	validation_1-mlogloss:0.947267
# [4]	validation_0-mlogloss:0.769553	validation_1-mlogloss:0.871933
# [5]	validation_0-mlogloss:0.70487	validation_1-mlogloss:0.81453
# [6]	validation_0-mlogloss:0.654573	validation_1-mlogloss:0.770619
# [7]	validation_0-mlogloss:0.611974	validation_1-mlogloss:0.731794
# [8]	validation_0-mlogloss:0.575798	validation_1-mlogloss:0.70103
# [9]	validation_0-mlogloss:0.541659	validation_1-mlogloss:0.671469
# [10]	validation_0-mlogloss:0.513218	validation_1-mlogloss:0.648299
# [11]	validation_0-mlogloss:0.488117	validation_1-mlogloss:0.627648
# [12]	validation_0-mlogloss:0.465995	validation_1-mlogloss:0.609641
# [13]	validation_0-mlogloss:0.446337	validation_1-mlogloss:0.594298
# [14]	validation_0-mlogloss:0.427126	validation_1-mlogloss:0.579995
# [15]	validation_0-mlogloss:0.409398	validation_1-mlogloss:0.565011
# [16]	validation_0-mlogloss:0.39401	validation_1-mlogloss:0.553054
# [17]	validation_0-mlogloss:0.379188	validation_1-mlogloss:0.543022
# [18]	validation_0-mlogloss:0.365207	validation_1-mlogloss:0.532529
# [19]	validation_0-mlogloss:0.352409	validation_1-mlogloss:0.523778
# [20]	validation_0-mlogloss:0.338585	validation_1-mlogloss:0.512741
# [21]	validation_0-mlogloss:0.327487	validation_1-mlogloss:0.504671
# [22]	validation_0-mlogloss:0.316103	validation_1-mlogloss:0.496792
# [23]	validation_0-mlogloss:0.305784	validation_1-mlogloss:0.489347
# [24]	validation_0-mlogloss:0.295163	validation_1-mlogloss:0.481345
# [25]	validation_0-mlogloss:0.286068	validation_1-mlogloss:0.476022
# [26]	validation_0-mlogloss:0.277075	validation_1-mlogloss:0.471218
# [27]	validation_0-mlogloss:0.26944	validation_1-mlogloss:0.467092
# [28]	validation_0-mlogloss:0.261542	validation_1-mlogloss:0.461821
# [29]	validation_0-mlogloss:0.25453	validation_1-mlogloss:0.457363
# [30]	validation_0-mlogloss:0.248127	validation_1-mlogloss:0.452604
# [31]	validation_0-mlogloss:0.241481	validation_1-mlogloss:0.449287
# [32]	validation_0-mlogloss:0.235515	validation_1-mlogloss:0.445914
# [33]	validation_0-mlogloss:0.229271	validation_1-mlogloss:0.443014
# [34]	validation_0-mlogloss:0.223148	validation_1-mlogloss:0.439982
# [35]	validation_0-mlogloss:0.217019	validation_1-mlogloss:0.436412
# [36]	validation_0-mlogloss:0.212211	validation_1-mlogloss:0.433004
# [37]	validation_0-mlogloss:0.207205	validation_1-mlogloss:0.429732
# [38]	validation_0-mlogloss:0.20224	validation_1-mlogloss:0.427076
# [39]	validation_0-mlogloss:0.197901	validation_1-mlogloss:0.425025
# [40]	validation_0-mlogloss:0.193508	validation_1-mlogloss:0.422974
# [41]	validation_0-mlogloss:0.188707	validation_1-mlogloss:0.420019
# [42]	validation_0-mlogloss:0.185074	validation_1-mlogloss:0.418258
# [43]	validation_0-mlogloss:0.181012	validation_1-mlogloss:0.416381
# [44]	validation_0-mlogloss:0.1775	validation_1-mlogloss:0.414432
# [45]	validation_0-mlogloss:0.172775	validation_1-mlogloss:0.411582
# [46]	validation_0-mlogloss:0.168458	validation_1-mlogloss:0.410005
# [47]	validation_0-mlogloss:0.165472	validation_1-mlogloss:0.408252
# [48]	validation_0-mlogloss:0.162152	validation_1-mlogloss:0.406243
# [49]	validation_0-mlogloss:0.158504	validation_1-mlogloss:0.404675
# [50]	validation_0-mlogloss:0.155239	validation_1-mlogloss:0.403241
# [51]	validation_0-mlogloss:0.151542	validation_1-mlogloss:0.401818
# [52]	validation_0-mlogloss:0.148474	validation_1-mlogloss:0.400013
# [53]	validation_0-mlogloss:0.145551	validation_1-mlogloss:0.398021
# [54]	validation_0-mlogloss:0.142642	validation_1-mlogloss:0.396451
# [55]	validation_0-mlogloss:0.13949	validation_1-mlogloss:0.395165
# [56]	validation_0-mlogloss:0.136055	validation_1-mlogloss:0.393873
# [57]	validation_0-mlogloss:0.133006	validation_1-mlogloss:0.392826
# [58]	validation_0-mlogloss:0.130006	validation_1-mlogloss:0.390995
# [59]	validation_0-mlogloss:0.127192	validation_1-mlogloss:0.389298
# [60]	validation_0-mlogloss:0.124724	validation_1-mlogloss:0.388135
# [61]	validation_0-mlogloss:0.122204	validation_1-mlogloss:0.386889
# [62]	validation_0-mlogloss:0.119854	validation_1-mlogloss:0.385677
# [63]	validation_0-mlogloss:0.117321	validation_1-mlogloss:0.384371
# [64]	validation_0-mlogloss:0.11463	validation_1-mlogloss:0.383008
# [65]	validation_0-mlogloss:0.11246	validation_1-mlogloss:0.381732
# [66]	validation_0-mlogloss:0.109808	validation_1-mlogloss:0.380911
# [67]	validation_0-mlogloss:0.107619	validation_1-mlogloss:0.380112
# [68]	validation_0-mlogloss:0.105569	validation_1-mlogloss:0.379172
# [69]	validation_0-mlogloss:0.103921	validation_1-mlogloss:0.378704
# [70]	validation_0-mlogloss:0.102176	validation_1-mlogloss:0.378238
# [71]	validation_0-mlogloss:0.100406	validation_1-mlogloss:0.377299
# [72]	validation_0-mlogloss:0.098404	validation_1-mlogloss:0.376137
# [73]	validation_0-mlogloss:0.096775	validation_1-mlogloss:0.375678
# [74]	validation_0-mlogloss:0.09466	validation_1-mlogloss:0.374849
# [75]	validation_0-mlogloss:0.092855	validation_1-mlogloss:0.374072
# [76]	validation_0-mlogloss:0.09047	validation_1-mlogloss:0.37318
# [77]	validation_0-mlogloss:0.088724	validation_1-mlogloss:0.37189
# [78]	validation_0-mlogloss:0.087079	validation_1-mlogloss:0.37065
# [79]	validation_0-mlogloss:0.085766	validation_1-mlogloss:0.370373
# [80]	validation_0-mlogloss:0.08396	validation_1-mlogloss:0.369424
# [81]	validation_0-mlogloss:0.082266	validation_1-mlogloss:0.368483
# [82]	validation_0-mlogloss:0.081023	validation_1-mlogloss:0.36804
# [83]	validation_0-mlogloss:0.079635	validation_1-mlogloss:0.367469
# [84]	validation_0-mlogloss:0.078346	validation_1-mlogloss:0.366732
# [85]	validation_0-mlogloss:0.076979	validation_1-mlogloss:0.366909
# [86]	validation_0-mlogloss:0.075383	validation_1-mlogloss:0.366515
# [87]	validation_0-mlogloss:0.074199	validation_1-mlogloss:0.365853
# [88]	validation_0-mlogloss:0.073003	validation_1-mlogloss:0.365761
# [89]	validation_0-mlogloss:0.071755	validation_1-mlogloss:0.365479
# [90]	validation_0-mlogloss:0.070637	validation_1-mlogloss:0.365324
# [91]	validation_0-mlogloss:0.069209	validation_1-mlogloss:0.365112
# [92]	validation_0-mlogloss:0.067818	validation_1-mlogloss:0.365093
# [93]	validation_0-mlogloss:0.066907	validation_1-mlogloss:0.364693
# [94]	validation_0-mlogloss:0.065898	validation_1-mlogloss:0.364101
# [95]	validation_0-mlogloss:0.064902	validation_1-mlogloss:0.363795
# [96]	validation_0-mlogloss:0.063925	validation_1-mlogloss:0.363508
# [97]	validation_0-mlogloss:0.062687	validation_1-mlogloss:0.363139
# [98]	validation_0-mlogloss:0.06159	validation_1-mlogloss:0.362836
# [99]	validation_0-mlogloss:0.060426	validation_1-mlogloss:0.363203
# ============================================================
#               precision    recall  f1-score   support
#
#            0       0.86      0.93      0.89      3631
#            1       0.81      0.82      0.82       811
#            2       0.84      0.72      0.77       635
#            3       0.74      0.69      0.71       495
#            4       0.83      0.76      0.79       314
#            5       0.99      0.94      0.96       367
#            6       0.97      0.95      0.96       772
#            7       0.82      0.73      0.77       531
#            8       0.93      0.95      0.94      1018
#            9       0.79      0.68      0.73       431
#           10       0.78      0.58      0.67        74
#           11       0.87      0.80      0.84       380
#           12       0.95      0.96      0.96      1102
#           13       0.92      0.93      0.92       300
#
#     accuracy                           0.87     10861
#    macro avg       0.86      0.82      0.84     10861
# weighted avg       0.87      0.87      0.87     10861
#
# [[3389   63   39   17   16    1    3   36    0   27    0   17   23    0]
#  [  90  664    2   11   10    0    4    2   14    6    0    2    6    0]
#  [  96   11  455   12    1    1    0   15    8    7    0    7    9   13]
#  [  64   22    7  343    5    0    3    1   28   11    5    2    2    2]
#  [  24   10    4   15  239    0    1    2    3    8    0    1    2    5]
#  [  11    3    0    0    0  344    4    2    0    3    0    0    0    0]
#  [  11    7    2    1    0    1  731   13    1    3    0    0    1    1]
#  [  99   12    9    2    3    1    2  389    2    4    0    2    5    1]
#  [   2    8    8   16    0    0    1    0  968    1    2    9    0    3]
#  [  88    8    4   17    7    1    0    3    4  294    2    1    2    0]
#  [   1    0    0   18    1    0    0    0    5    4   43    2    0    0]
#  [  41    5    2   10    0    0    1    0    8    3    3  305    2    0]
#  [  28    4    1    0    0    0    0   10    0    3    0    1 1055    0]
#  [   0    0    7    3    7    0    3    0    2    0    0    0    0  278]]
# Accuracy: 87.44%
