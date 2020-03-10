import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

data_train = pd.read_csv('data_train_color_test_normilized.csv')
X_train, y_train = data_train.drop('type_leaf', axis=1), data_train.type_leaf

data_validation = pd.read_csv('data_valid_color_test_normilized.csv')
X_val, y_val = data_validation.drop('type_leaf', axis=1), data_validation.type_leaf

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
parameters = {'max_features': [2, 5, 7, 14, 20],
              'min_samples_leaf': [7, 14, 30],
              'max_depth': [2, 10, 30, 60],
              'min_samples_split': [2, 4, 7],
              'n_estimators': [20, 40, 50, 100]}
rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X_train, y_train)
print(gcv.best_estimator_)
print(gcv.best_score_)
print(gcv.best_params_)

# Result:
# 0.8430940111538154
# {'max_depth': 60, 'max_features': 14, 'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 100}
# RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=60, max_features=14,
#                        max_leaf_nodes=None, max_samples=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=2, min_samples_split=7,
#                        min_weight_fraction_leaf=0.0, n_estimators=100,
#                        n_jobs=-1, oob_score=False, random_state=42, verbose=0,
#                        warm_start=False)
