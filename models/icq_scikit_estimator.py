import sys
import os
sys.path.append(os.path.abspath('../helpers'))

from helpers.database_helpers import replicate_classes
from helpers.icq_methods import update_weights
from helpers.plot_graphs import plot_graph

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class IcqClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, 
                 classifier_function, 
                 rate_succ=0.2, 
                 sigma_q_weights=None, 
                 max_iter=1000, 
                 reset_weights=1000,
                 random_seed=1,
                 learning_rate=0.01,
                 plot_graphs_and_metrics = True):
        self.rate_succ = rate_succ
        self.classifier_function = classifier_function
        self.max_iter = max_iter
        self.sigma_q_weights = sigma_q_weights
        self.reset_weights = reset_weights
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.plot_graphs_and_metrics = plot_graphs_and_metrics

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        
        X,y = replicate_classes(X, y)
            
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        low  = -1
        high = 1
        dimensions = len(X[0])

        rate = 0
        
        weight = np.random.uniform(low=low, high=high, size=(dimensions,))
        
        ITERATION = 0
        
        bestWeight = []
        bestRate = 0.0
        self.ratesDuringTraining = []
        
        while rate < self.rate_succ and ITERATION < self.max_iter:
            rate = 0
            # training step
            for x_train, y_train in zip(X, y):
                z, p_cog, _ = self.classifier_function(x_train, weight, self.sigma_q_weights)
                weight = update_weights(weight, y_train, z, x_train, p_cog, n=self.learning_rate)
                
            for x_train, y_train in zip(X, y):
                z, p_cog, _ = self.classifier_function(x_train, weight, self.sigma_q_weights)            
                if z == y_train:
                    rate +=1
            
            rate = rate/len(X)
            self.ratesDuringTraining.append(rate)
            ITERATION += 1
            if (rate >= bestRate):
                bestWeight = weight
                bestRate = rate
            if ITERATION % self.reset_weights == 0:
                weight = np.random.uniform(low=low, high=high, size=(dimensions,))
            
        self.rate = bestRate
        self.weight_ = bestWeight
        self.X_ = X
        self.y_ = y
        
        if self.plot_graphs_and_metrics:
            print("best weight", bestWeight)
            print("best rate", bestRate)
            plot_graph(range(ITERATION), self.ratesDuringTraining , "iter", "rate")
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'weight_'])

        # Input validation
        X = check_array(X)
        
        outputs = []
        for x in X:                   
            z, _, _ = self.classifier_function(x, self.weight_, self.sigma_q_weights)
            outputs.append(z)
                               
        return outputs

    def predict_proba(self, X):
        outputs = []
        for x in X:                   
            _, p_cog, _ = self.classifier_function(x, self.weight_, self.sigma_q_weights)
            outputs.append([1-p_cog.real, p_cog.real])
                               
        return np.array(outputs)