import math
import numpy as np

from engine.expression import Expression

class Activation(Expression):
    mode: str


class ScalarActivation(Activation):
    def __init__(self, x: Expression):
        self.x = x

class ReLU(ScalarActivation):
    def eval(self):
        self.x.eval()
        self.value = max(0, self.x.value)

    def derive(self, seed: float):
        if self.x.value > 0:
            self.x.derive(seed)


class Sigmoid(ScalarActivation):
    def eval(self):
        self.x.eval()
        self.value = 1 / (1 + math.exp(-self.x.value))

    def derive(self, seed: float):
        self.x.derive(seed * self.value * (1 - self.value))


class VectorActivation(Activation):
    def __init__(self, x: list[Expression], index: int):
        self.x = x
        self.index = index


class Softmax(VectorActivation):
    def eval(self):
        for xi in self.x:
            xi.eval()
        logits = np.array([xi.value for xi in self.x])
        exps = np.exp(logits - logits.max())  # stability
        self.s = exps / exps.sum()
        self.value = self.s[self.index]

    def derive(self, seed):
        # seed is dL/ds_i
        i = self.index
        s = self.s

        for k, xk in enumerate(self.x):
            if k == i:
                grad = seed * s[i] * (1 - s[i])
            else:
                grad = seed * (-s[i] * s[k])

            xk.derive(grad)