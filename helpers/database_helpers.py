import numpy as np

import numpy as np
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, classification_report
from sklearn import datasets

def replicate_classes(X, y):

    X_class_minoritary = X[y==1]
    y_class_minoritary = y[y==1]
    
    conX = np.concatenate((X, X_class_minoritary), axis=0)
    
    conY = np.concatenate((y, y_class_minoritary), axis=0)
    
    return conX, conY


def get_iris():
    iris = datasets.load_iris()
    X = iris.data[:, [0,1,2, 3]]
    y = iris.target
    return X, y

def get_stratified_kfold(k_folds=10, random_seed=42):
    return StratifiedKFold(n_splits=k_folds, random_state=random_seed, shuffle=True)