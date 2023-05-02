import sys
import os
sys.path.append(os.path.abspath('../helpers'))

from helpers.icq_executions import execute_classifier_original_normal_sigma_q, execute_classifier_split_input_weight_normal_sigma_q

from icq_scikit_estimator import IcqClassifier

RANDOM_SEED = 42

def OriginalClassifier(max_iter = 1000, success_rate=1.0, learning_rate=0.01, plot_graphs_and_metrics=True):
    """
        Returns a Scikit-Learn classifier that uses the original ICQ classifier from the Interactive Quantum Classifier Inspired by Quantum Open System Theory article.

        Always use random seed as icq_estimators.RANDOM_SEED (default 42) and never reset weights.
    """
    return IcqClassifier(
            classifier_function=execute_classifier_original_normal_sigma_q, 
            sigma_q_weights = [1,1,1,0],
            max_iter=max_iter,
            accuracy_succ=success_rate,
            learning_rate=learning_rate,
            plot_graphs_and_metrics=plot_graphs_and_metrics,
            random_seed=RANDOM_SEED,
            reset_weights=0)

def InputsOnEnvClassifier(max_iter = 1000, success_rate=1.0, sigma_q_weights=[1,1,1,0]):
    """
        Returns a Scikit-Learn classifier that uses the modified version of ICQ classifier from the Interactive Quantum Classifier Inspired by Quantum Open System Theory article. See ../helpers/icq_methods.create_and_execute_classifier_new_approach for more information.

        Always use random seed as icq_estimators.RANDOM_SEED (default 42) and never reset weights.
    """
    return IcqClassifier(
            classifier_function=execute_classifier_split_input_weight_normal_sigma_q, 
            sigma_q_weights = sigma_q_weights,
            max_iter=max_iter,
            accuracy_succ=success_rate)