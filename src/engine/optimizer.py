import numpy as np


class Optimizer:
    def step(self, weights):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float, clip_gradients: float = None):
        self.learning_rate = learning_rate
        self.clip_gradients = clip_gradients

    def step(self, weights):
        def update_weight(weight):
            grad = np.clip(weight.grad, a_min=-self.clip_gradients, a_max=self.clip_gradients) if self.clip_gradients else weight.grad
            weight.value -= self.learning_rate * grad
            weight.grad = 0.0

        vec_update_weights = np.vectorize(update_weight)
        for layer_weights in weights:
            vec_update_weights(layer_weights)
