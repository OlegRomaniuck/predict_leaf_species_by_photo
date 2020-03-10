import pickle
import itertools
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    plt.figure()
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')

    plt.show()


data_validation = pd.read_csv('data_valid_color_test_normilized.csv')

X_val, y_val = data_validation.drop('type_leaf', axis=1), data_validation.type_leaf

filename = 'randomforestclassl.sav'
model = pickle.load(open(filename, "rb"))
predictions = model.predict(X_val)

print("Accuracy:", metrics.accuracy_score(y_val, predictions))

print("Training F1 Micro Average: {}".format(f1_score(y_val, predictions, average="micro")))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
results = cross_val_score(model, X_val, y_val, cv=skf)
# evaluate accuracy on validation data set
print("CV accuracy score: {:.2f}%".format(results.mean() * 100))

print(metrics.classification_report(y_val, predictions))
pd = pd.crosstab(y_val, predictions, rownames=['Actual Species'], colnames=['Predicted Species'])
class_names = ['Tomato', ' Grape', 'Apple', 'Pepper', 'Strawberry', 'Squash', 'Corn', 'Peach', 'Soybean', 'Potato',
               'Raspberry', 'Cherry', 'Orange', 'Blueberry']
matrix = confusion_matrix(y_val, predictions)
plot_confusion_matrix(matrix, class_names, title='Confusion matrix, without normalization')

# Accuracy: 0.8536967130098517
# Training F1 Micro Average: 0.8536967130098517
# CV accuracy score: 79.38%


# Accuracy: 0.8123324556565634323
# Training F1 Micro Average: 0.812434190837347437843
# CV accuracy score: 79.38%


#               precision    recall  f1-score   support
#
#            Tomato          0.81      0.95      0.88      3631
#            Grape           0.81      0.76      0.78       811
#            Apple           0.90      0.60      0.72       635
#            Pepper          0.77      0.62      0.69       495
#            Strawberry      0.80      0.82      0.81       314
#            Squash          0.94      0.90      0.92       367
#            Corn            0.97      0.93      0.95       772
#            Peach           0.85      0.66      0.75       531
#            Soybean         0.88      0.95      0.91      1018
#            Potato          0.85      0.59      0.70       431
#            Raspberry       0.93      0.58      0.72        74
#            Cherry          0.88      0.74      0.80       380
#            Orange          0.93      0.94      0.94      1102
#            Blueberry       0.86      0.91      0.88       300
# #
#     accuracy                           0.85     10861
#    macro avg       0.87      0.78      0.82     10861
# weighted avg       0.86      0.85      0.85     10861

# Predicted Species    0    1    2    3    4    5   ...   8    9   10   11    12   13
# Actual Species                                    ...
# 0                  3457   46    8    9   16    8  ...    0   20   0   14    29    0
# 1                   123  615    0    9   15    2  ...   28    5   0    1    10    0
# 2                   165   13  379   12    2    0  ...   14    2   0    3    13   17
# 3                    81   21    6  309   10    1  ...   43    5   1    0     4   10
# 4                    25   10    3    5  256    0  ...    2    3   0    2     0    4
# 5                    22    5    0    0    0  329  ...    0    3   0    0     0    0
# 6                    18    8    1    1    0    5  ...    1    2   0    1     2    5
# 7                   127   15    6    2    2    3  ...    3    4   0    1    12    3
# 8                     2    5    5   18    0    0  ...  970    1   1   11     0    5
# 9                   120   10    0   14   11    3  ...   10  256   0    2     2    0
# 10                    2    0    0   11    2    0  ...   11    0  43    3     0    2
# 11                   60    5    2    9    0    0  ...   16    1   1  280     5    0
# 12                   51    6    1    0    0    0  ...    0    0   0    0  1036    0
# 13                    0    0    8    1    7    0  ...    7    0   0    0     0  274
