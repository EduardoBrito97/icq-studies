import numpy as np
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from sklearn import datasets

import sys
import os
sys.path.append(os.path.abspath('../models'))
from models.icq_scikit_estimator import IcqClassifier

def executeIris(random_seed = 1, 
                classifier_function=None, 
                sigma_q_weights=[1,1,1,0], 
                one_vs_classifier=OneVsRestClassifier, 
                max_iter=3000):
    """
        Executes ICQ classifier against Iris dataset (loaded from scikit.datasets) using classifier_function as classifier (either ../helpers/icq_methods.create_and_execute_classifier or ../helpers/icq_methods.create_and_execute_classifier_new_approach.)
    """
    # load dataset
    iris = datasets.load_iris()
    X = iris.data[:, [0,1,2, 3]]
    y = iris.target

    # split training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = random_seed, stratify = y)

    normalized_X_train = normalize(X_train)
    normalized_X_test  = normalize(X_test)
    
    clf = one_vs_classifier(
        IcqClassifier(
            classifier_function=classifier_function, 
            sigma_q_weights = sigma_q_weights,
            max_iter=max_iter,
            rate_succ=1.0)).fit(normalized_X_train, y_train)

    score = clf.score(normalized_X_test, y_test)
    f1score = f1_score(clf.predict(normalized_X_test), y_test, average='macro')

    return score, f1score

def executeIrisOneVsRest(random_seed = 1, classifier_function=None, sigma_q_weights=[1,1,1,0], max_iter=3000):
    return executeIris(random_seed, classifier_function, sigma_q_weights, OneVsRestClassifier, max_iter)

def executeIrisOneVsOne(random_seed = 1, classifier_function=None, sigma_q_weights=[1,1,1,0], max_iter=3000):
    return executeIris(random_seed, classifier_function, sigma_q_weights, OneVsOneClassifier, max_iter)

def print_metrics(scores, f1scores):
    print("Scores:", scores)
    print("Best score:", np.max(scores))
    print("F1-Scores:", f1scores)
    print("Max F1-Score:", np.max(f1scores))