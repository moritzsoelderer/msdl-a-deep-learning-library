import numpy as np
import math

from engine.expression import Expression
from engine.array_utils import map_to_value, vec_eval

class Loss(Expression):
    def __init__(self, logits, y_true):
        pass

class CrossEntropyWithLogits(Loss):
    def __init__(self, logits, y_true):
        self.logits = logits
        self.y_true = np.array(y_true, dtype=float)

    def eval(self):
        vec_eval(self.logits)
        z_vals = map_to_value(self.logits)

        # Stable softmax
        max_logit = max(logit.value for logit in self.logits)
        exps = [math.exp(logit.value - max_logit) for logit in self.logits]
        self.s = [e / sum(exps) for e in exps]

        # Cross entropy loss: -sum(y * log(s))
        # (only non-zero at true class)
        true_idx = np.argmax(self.y_true)
        self.value = -np.log(self.s[true_idx] + 1e-12)

    def derive(self, seed):
        # seed is dL/dLoss, usually 1
        s = self.s

        for k, z in enumerate(self.logits):
            grad = seed * (s[k] - self.y_true[k])
            z.derive(grad)