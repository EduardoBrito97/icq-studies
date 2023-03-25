import sys
import os
import time
sys.path.append(os.path.abspath('../helpers'))

from helpers.database_helpers import replicate_classes
from helpers.icq_methods import update_weights, update_batched_weights
from helpers.plot_graphs import plot_graph

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class IcqClassifier(ClassifierMixin, BaseEstimator):
    """
        Returns an Scikit-Learn based estimator that uses ICQ classificator (https://ieeexplore.ieee.org/document/9533917) to classify instances.

        It estimates only binary classifications. For multi-class problems, you can use e.g. sklearn.multiclass.OneVsOneClassifier or sklearn.multiclass.OneVsRestClassifier.

        Attributes:
            classifier_function (fun): check /helpers/icq_executions.py file to see available functions

            accuracy_succ (float): accuracy considered as successful training.

            sigma_q_weights (4 sized array): weights for sigma Q sum. See ../helpers/icq_methods.get_weighted_sigmaQ for more info.

            max_iter (int): max number of training epochs.

            reset_weights_epoch (int): max amount of epochs that a random weight should be trained. If reached, it will reset the weights to random numbers again and will keep training. If set to 0, it will never be reset.

            learning_rate (float): weights' learning accuracy.

            plot_graphs_and_metrics (boolean): prints training best weights, accuracy and epoch x accuracy graph.

            do_classes_refit (boolean): resamples classes in order to have same amount of 0s and 1s instances. See ../helpers/database_helpers.replicate_classes

            batch (integer): batch size used during training.

            accuracys_during_training_ (array): accuracy throughout the training.

            X_ (array of arrays): instances attributes used for training.

            Y_ (array): instances classes used for training.

            weight_ (array): best weights from training.

            accuracy_ (float): best accuracy from training.
    """
    def __init__(self, 
                 classifier_function, 
                 accuracy_succ=0.2, 
                 sigma_q_weights=None, 
                 max_iter=1000, 
                 reset_weights_epoch=0,
                 random_seed=1,
                 learning_rate=0.01,
                 plot_graphs_and_metrics=True,
                 do_classes_refit=True,
                 batch=1):
        self.accuracy_succ = accuracy_succ
        self.classifier_function = classifier_function
        self.max_iter = max_iter
        self.sigma_q_weights = sigma_q_weights
        self.reset_weights_epoch = reset_weights_epoch
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.plot_graphs_and_metrics = plot_graphs_and_metrics
        self.do_classes_refit=do_classes_refit
        self.batch = batch

    def fit(self, X, y):
        """
            Trains the ICQ classifier using X as instances attributes and y as instances classes.

            To have a fair training, it replicates the minority class to have the same number of instances as the majority class. See ../helpers/database_helpers.replicate_classes for more info or to change the replication approach.

            X: N x M matrix, where M is the number of attributes and N is the number of instances.
            y: N sized array of 0s or 1s values, where N is the number of instances.

            Returns the trained classifier.
        """
        # Replicates classes to have same number of 0s and 1s examples
        if (self.do_classes_refit):
            X,y = replicate_classes(X, y, self.random_seed)
            
        # Check that X and y have correct shape (i.e. same amount of examples)
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Creates weights based on a [-1, 1] uniform distribution
        low = -1
        high = 1
        dimensions = len(X[0])
        num_of_instances = len(X)
        
        # Setting random seed to have always same result
        np.random.seed(self.random_seed)
        weight = np.random.uniform(low=low, high=high, size=(dimensions,))
        
        ITERATION = 0
        best_weight = []
        best_accuracy = 0.0
        accuracy = 0
        self.accuracy_during_training_ = []
        
        # Executing the training itself
        while accuracy < self.accuracy_succ and ITERATION < self.max_iter:
            accuracy = 0
            accumulated_loss = np.zeros((dimensions))
            
            # Training step
            for i, (x_train, y_train) in enumerate(zip(X, y)):
                # Execute the classifier with the weights we have now...
                z, p_cog, _ = self.classifier_function(vector_x=x_train, vector_w=weight, sigma_q_params=self.sigma_q_weights)

                accumulated_loss += (z - y_train) * x_train
                if self.batch <= 1:
                    weight = update_weights(weight, y_train, z, x_train, p_cog, n=self.learning_rate)
                elif i % self.batch == 0 or i == num_of_instances - 1:
                    weight = update_batched_weights(weight, accumulated_loss/self.batch, self.learning_rate)
                    accumulated_loss = np.zeros((dimensions))
                
            # After executing everything and updating the weights for the whole set example, we compute current accuracy
            for x_train, y_train in zip(X, y):
                # Classify using current weight...
                z, p_cog, _ = self.classifier_function(vector_x=x_train, vector_w=weight, sigma_q_params=self.sigma_q_weights)            
                
                # ... and checks if we got it right
                if z == y_train:
                    accuracy +=1
            
            # Computing actual accuracy...
            accuracy = accuracy/len(y)
            self.accuracy_during_training_.append(accuracy)
            ITERATION += 1

            # ... and checking if this is the best one so far
            if (accuracy >= best_accuracy):
                best_weight = weight
                best_accuracy = accuracy

            # In case reset weights is defined 
            if self.reset_weights_epoch != 0 and ITERATION % self.reset_weights_epoch == 0:
                np.random.seed(self.random_seed)
                weight = np.random.uniform(low=low, high=high, size=(dimensions,))
            
        self.accuracy_ = best_accuracy
        self.weight_ = best_weight
        self.X_ = X
        self.y_ = y
        
        if self.plot_graphs_and_metrics:
            print("best weight", best_weight)
            print("best accuracy", best_accuracy)
            plot_graph(range(ITERATION), self.accuracy_during_training_ , "epoch", "accuracy")
        
        # Return the classifier
        return self

    def predict(self, X):
        """
            Returns the predicted class for each X instance - either 0 or 1.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'weight_'])

        # Input validation
        X = check_array(X)
        
        # Classifies each instance
        outputs = []
        for x in X:                   
            z, _, _ = self.classifier_function(vector_x=x, vector_w=self.weight_, sigma_q_params=self.sigma_q_weights)
            outputs.append(z)

        # Returns either 0 or 1      
        return outputs

    def predict_proba(self, X):
        """
            Returns the probability of each instance being of each class - either 0 or 1.
        """
        outputs = []
        for x in X:                   
            _, p_cog, _ = self.classifier_function(vector_x=x, vector_w=self.weight_, sigma_q_params=self.sigma_q_weights)
            outputs.append([1-p_cog.real, p_cog.real])

        # Returns the probability of being either 0 or 1           
        return np.array(outputs)