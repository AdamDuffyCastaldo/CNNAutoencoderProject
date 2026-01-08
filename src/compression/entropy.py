import numpy as np


def entropy(array):
    array = array[array > 0]
    array = (lambda x : x * np.log2(x))(array)
    entropy = -sum(array)
    return entropy
