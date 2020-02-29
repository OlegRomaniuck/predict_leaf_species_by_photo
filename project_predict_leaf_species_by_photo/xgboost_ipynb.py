
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data_train = pd.read_csv('/leaf_features_color_train.csv')
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

data_test = pd.read_csv('leaf_features_color_test.csv')
print(data_test.shape)
data_test["type_leaf"] = data_test["type_leaf"] - 1
data_test.drop("Unnamed: 0", axis=1, inplace=True)
data_test.drop("name", axis=1, inplace=True)


X_test, y_test = data_test.drop('type_leaf', axis =1), data_test.type_leaf
print(X_test.shape)
print(y_test.shape)



dtrain = xgb.DMatrix(data=X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(data=X_test)
eval_set=[(dvalid, 'eval')]
params = {
    'max_depth': 10,
    'objective': 'multi:softmax',  # error evaluation for multiclass training
    'num_class': 14,
    'n_gpus': 0
}

bst = xgb.train(params, dtrain, evals=eval_set)

pred = bst.predict(dtest)


print(classification_report(y_test, pred))


cm = confusion_matrix(y_test, pred)
print(cm)

def plot_confusion_matrix(cm, classes, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
        plt.savefig('confusion-matrix.png')

plot_confusion_matrix(cm, ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7), (8)', '(9)', '(10)', '(11)', '(12)', '(13), (14)'])
pickle.dump(bst, open("xgboost_based_ipynb.dat", "wb"))
