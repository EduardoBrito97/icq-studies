import numpy as np
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, classification_report
from sklearn import datasets
from imblearn.over_sampling import RandomOverSampler

def replicate_classes(X, y, random_seed=42):
    ones_indices = y == 1
    ones = len(y[ones_indices])

    zeros_indices = y == 0
    zeros = len(y[zeros_indices])

    # If the num of majority class is a multiple of the minority one, we just need to replicate
    if (zeros % ones == 0 or ones % zeros == 0):
        if (zeros > ones):
            # If we have more zeros than ones, we need to replicate the ones...
            indices_to_replicate = ones_indices
            # ... and replicate num of zeros / num of ones times
            num_of_replications = int(zeros / ones)
        else:
            # But if we have more ones than zeros, we have the opposite logic
            indices_to_replicate = zeros_indices
            num_of_replications = int(ones / zeros)

        # We don't need to care if there are the same number of zeros and ones because in this case we'd have range(0) - which won't execute the for loop
        for _ in range(num_of_replications - 1):
            conX = np.concatenate((X, X[indices_to_replicate]), axis=0)
            conY = np.concatenate((y, y[indices_to_replicate]), axis=0)
            indices_to_replicate = np.append(indices_to_replicate, [False for _ in range(len(X) - len(indices_to_replicate)) ])
    else:
        # In case we don't have a well behaved dataset, we just randomly replicate the minority class
        ros = RandomOverSampler(random_state=random_seed)
        conX, conY = ros.fit_resample(X, y)
    
    return conX, conY


def get_iris():
    iris = datasets.load_iris()
    X = iris.data[:, [0,1,2,3]]
    y = iris.target
    return X, y

def get_wine():
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    return X, y

def get_stratified_kfold(k_folds=10, random_seed=42):
    return StratifiedKFold(n_splits=k_folds, random_state=random_seed, shuffle=True)