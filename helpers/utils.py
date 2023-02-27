import numpy as np
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, classification_report
from sklearn import datasets

import sys
import os
sys.path.append(os.path.abspath('../models'))
from models.icq_scikit_estimator import IcqClassifier

K_FOLDS=10

def executeIris(random_seed = 1, 
                classifier_function=None, 
                sigma_q_weights=[1,1,1,0], 
                one_vs_classifier=OneVsRestClassifier, 
                max_iter=3000,
                plot_graphs_in_classifier=False,
                print_each_fold_metric=True,
                print_avg_metric=True):
    """
        Executes ICQ classifier against Iris dataset (loaded from scikit.datasets) using classifier_function as classifier (either ../helpers/icq_methods.create_and_execute_classifier or ../helpers/icq_methods.create_and_execute_classifier_new_approach.)
    """
    # load dataset
    iris = datasets.load_iris()
    X = iris.data[:, [0,1,2, 3]]
    y = iris.target

    # split training set and test set
    skf = StratifiedKFold(n_splits=K_FOLDS, random_state=random_seed, shuffle=True)

    scores = []
    f1scores = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        normalized_X_train = normalize(X_train)
        normalized_X_test  = normalize(X_test)

        clf = one_vs_classifier(
            IcqClassifier(
                classifier_function=classifier_function, 
                sigma_q_weights=sigma_q_weights,
                max_iter=max_iter,
                accuracy_succ=1.0,
                plot_graphs_and_metrics=plot_graphs_in_classifier,
                random_seed=random_seed)).fit(normalized_X_train, y_train)

        score = clf.score(normalized_X_test, y_test)
        f1score = f1_score(clf.predict(normalized_X_test), y_test, average='macro')

        scores.append(score)
        f1scores.append(f1score)

        if print_each_fold_metric:
            y_pred = clf.predict(X_test)
            print("K-Fold #" + str(i) + ":")
            print(classification_report(y_test, y_pred))
            print("-------------------------------------------------------------------------------------------------------------------")
    
    if print_avg_metric:
        print("AVG: Scores =", np.mean(scores), "F1-Scores =", np.mean(f1scores))
    return scores, f1scores

def executeIrisOneVsRest(random_seed=1, 
                        classifier_function=None, 
                        sigma_q_weights=[1,1,1,0], 
                        max_iter=3000,
                        print_each_fold_metric=True,
                        print_avg_metric=True):
    """
        Uses executeIris with sklearn.multiclass.OneVsRestClassifier
    """
    return executeIris(random_seed,
                       classifier_function,
                       sigma_q_weights,
                       OneVsRestClassifier,
                       max_iter,
                       print_each_fold_metric=print_each_fold_metric,
                       plot_graphs_in_classifier=False,
                       print_avg_metric=print_avg_metric)

def executeIrisOneVsOne(random_seed=1, 
                        classifier_function=None, 
                        sigma_q_weights=[1,1,1,0], 
                        max_iter=3000,
                        print_each_fold_metric=True,
                        print_avg_metric=True):
    """
        Uses executeIris with sklearn.multiclass.OneVsOneClassifier
    """
    return executeIris(random_seed,
                       classifier_function,
                       sigma_q_weights,
                       OneVsOneClassifier,
                       max_iter,
                       print_each_fold_metric=print_each_fold_metric,
                       plot_graphs_in_classifier=False,
                       print_avg_metric=print_avg_metric)

def print_metrics(scores, f1scores):
    print("Scores:", scores)
    print("Best score:", np.max(scores))
    print("F1-Scores:", f1scores)
    print("Max F1-Score:", np.max(f1scores))