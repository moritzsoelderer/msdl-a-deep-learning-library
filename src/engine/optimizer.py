import numpy as np

from engine.expression import Expression


class Optimizer:
    def step(self, weights):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, weights):
        def update_weight(weight):
            before = weight.value
            weight.value = weight.value - self.learning_rate * weight.grad
            #print(f"before {before}, grad {weight.grad}, after {weight.value}")
            weight.grad = 0.0

        vec_update_weights = np.vectorize(update_weight)
        for layer_weights in weights:
            vec_update_weights(layer_weights)
