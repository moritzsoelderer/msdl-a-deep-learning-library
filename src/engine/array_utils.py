import numpy as np
from engine.expression import Variable

def wrap_into_variables(arr: np.ndarray):
    return np.vectorize(lambda x: Variable(x))(arr)

def map_to_value(arr: np.ndarray):
    return np.vectorize(lambda x: x.value)(arr)

vec_eval = np.vectorize(lambda x: x.eval())
vec_derive = lambda seed: np.vectorize(lambda x: x.derive(seed))