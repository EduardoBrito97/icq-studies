import sys
import os
sys.path.append(os.path.abspath('../helpers'))

from helpers.icq_methods import create_and_execute_classifier, create_and_execute_classifier_new_approach

from icq_scikit_estimator import IcqClassifier

def OriginalClassifier(max_iter = 1000, success_rate=1.0):
    return IcqClassifier(
            classifier_function=create_and_execute_classifier, 
            sigma_q_weights = [1,1,1,0],
            max_iter=max_iter,
            rate_succ=success_rate)

def InputsOnEnvClassifier(max_iter = 1000, success_rate=1.0, sigma_q_weights=[1,1,1,0]):
    return IcqClassifier(
            classifier_function=create_and_execute_classifier_new_approach, 
            sigma_q_weights = sigma_q_weights,
            max_iter=max_iter,
            rate_succ=success_rate)