
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV


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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
parameters = {'max_features': [14], 'min_samples_leaf': [7, 14, 30], 'max_depth': [ 60]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X_train, y_train)

print(gcv.best_estimator_)
print(gcv.best_score_)
print(gcv.best_params_)


# 0.8430940111538154
# {'max_depth': 60, 'max_features': 14, 'min_samples_leaf': 2}
# RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=60, max_features=14,
#                        max_leaf_nodes=None, max_samples=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=7, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=100,
#                        n_jobs=-1, oob_score=False, random_state=42, verbose=0,
#                        warm_start=False)
# 0.8235416586114089
# {'max_depth': 60, 'max_features': 14, 'min_samples_leaf': 7}
