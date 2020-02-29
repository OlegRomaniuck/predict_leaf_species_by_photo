import pickle

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn import metrics


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split

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


rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=60, max_features=14, min_samples_leaf= 2, criterion = 'entropy')
#Train the model using the training sets y_pred=clf.predict(X_test)
rfc.fit(X_train,y_train)

# prediction on test set
y_pred=rfc.predict(X_val)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_val, y_pred))

print("Training F1 Micro Average: {}".format(f1_score(y_val, y_pred ,average = "micro")))

# Инициализируем стратифицированную разбивку нашего датасета для валидации
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# Обучаем на тренировочном датасете
results = cross_val_score(rfc, X_train, y_train, cv=skf)

# Оцениваем точность на тестовом датасете
print("CV accuracy score: {:.2f}%".format(results.mean()*100))


print(metrics.classification_report(y_val, y_pred))

# save the model to disk
filename = 'randomforestclassl.sav'
pickle.dump(rfc, open(filename, 'wb'))
# Accuracy: 0.8536967130098517
# Training F1 Micro Average: 0.8536967130098517
# CV accuracy score: 84.61%
#               precision    recall  f1-score   support
#
#            0       0.81      0.95      0.88      3631
#            1       0.81      0.76      0.78       811
#            2       0.90      0.60      0.72       635
#            3       0.77      0.62      0.69       495
#            4       0.80      0.82      0.81       314
#            5       0.94      0.90      0.92       367
#            6       0.97      0.93      0.95       772
#            7       0.85      0.66      0.75       531
#            8       0.88      0.95      0.91      1018
#            9       0.85      0.59      0.70       431
#           10       0.93      0.58      0.72        74
#           11       0.88      0.74      0.80       380
#           12       0.93      0.94      0.94      1102
#           13       0.86      0.91      0.88       300
#
#     accuracy                           0.85     10861
#    macro avg       0.87      0.78      0.82     10861
# weighted avg       0.86      0.85      0.85     10861
