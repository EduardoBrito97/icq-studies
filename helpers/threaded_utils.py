from sklearn.multiclass import OneVsRestClassifier
from threading import Thread

import time
import sys
import os
sys.path.append(os.path.abspath('../models'))
sys.path.append(os.path.abspath('../helpers'))
from database_helpers import get_iris
from utils import execute_model, print_metrics

NUM_OF_THREADS = 4

class ReturningThread(Thread):
    def run(self):
        try:
            if self._target:
                self._result = self._target(*self._args, **self._kwargs)
        finally:
            del self._target, self._args, self._kwargs

    def join(self):
        super().join()
        return self._result

def run_kfolds(classifier_function=None, 
                sigma_q_weights=[1, 1, 1], 
                one_vs_classifier=OneVsRestClassifier, 
                max_iter=3000,
                plot_graphs_in_classifier=False,
                print_each_fold_metric=True,
                print_avg_metric=True,
                learning_rate=0.01,
                accuracy_succ=1.00,
                dataset_load_method=get_iris,
                refit_db=True):
    scores = []
    f1scores = []
    ts = []
    start = time.process_time()
    for i_random_state in range(10, 70, 5):
        t = ReturningThread(target=execute_model, args=(i_random_state, classifier_function, sigma_q_weights, one_vs_classifier, max_iter, plot_graphs_in_classifier, print_each_fold_metric, print_avg_metric, learning_rate, accuracy_succ, dataset_load_method, refit_db))
        ts.append(t)
        t.start()
        if len(ts) >= NUM_OF_THREADS:
            curr_scores, curr_f1scores = ts[0].join()
            ts.remove(ts[0])
            print('Finished one thread after ', time.process_time() - start, "ms")
    
    for t in ts:
        curr_scores, curr_f1scores = t.join()
        scores.append(curr_scores)
        f1scores.append(curr_f1scores)
        print('Finished one thread after ', time.process_time() - start, "ms")
    
    print_metrics(scores, f1scores)
    return scores, f1scores