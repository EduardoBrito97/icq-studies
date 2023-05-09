import numpy as np
from imblearn.over_sampling import RandomOverSampler

def replicate_classes(X, y, random_seed=42):
    """
        Assumes that y is either 0 or 1, and tries to balance these two classes.
        
        If the amount of the majority class is a multiple of the minority, we completely replicate the minority until we reach the majority.

        In case it's not, we use imblear.over_sampling.RandomOverSampler with random_seed as seed to oversample and get the same amount of minority and majority class. See https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
    """
    ones_indices = y == 1
    ones = len(y[ones_indices])

    zeros_indices = y == 0
    zeros = len(y[zeros_indices])

    # If the num of majority class is a multiple of the minority one, we just need to replicate
    #if (zeros % ones == 0 or ones % zeros == 0):
    
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
    conX = X
    conY = y
    
    for _ in range(num_of_replications - 1):
        conX = np.concatenate((X, X[indices_to_replicate]), axis=0)
        conY = np.concatenate((y, y[indices_to_replicate]), axis=0)
        
    # In case we don't have a well behaved dataset, we get the database tuning attempt output and randomly replicate the minority class
    ros = RandomOverSampler(random_state=random_seed)
    conX, conY = ros.fit_resample(conX, conY)
    
    return conX, conY


