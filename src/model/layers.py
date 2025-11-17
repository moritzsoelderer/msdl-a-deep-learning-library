import numpy as np

from engine.expression import Constant, dot
from engine.array_utils import wrap_into_variables
from engine.activation import Activation, ScalarActivation, VectorActivation

class Layer():
    weights: np.ndarray

    def propagate(self, inputs) -> np.ndarray:
        return np.array([])


class LinearLayer(Layer):
    weights: np.ndarray
    activation: type[Activation] | None
    activation_mode: str

    def __init__(self, num_inputs: int, num_neurons: int, activation: type[Activation] | None = None) -> None:
        self.weights = wrap_into_variables(np.random.random_sample(size=(num_neurons, num_inputs)))
        self.activation = activation

    def propagate(self, inputs: np.ndarray) -> np.ndarray:
        outputs = []
        for sample in inputs:
            pre_acts = [dot(sample, neuron_weights) for neuron_weights in self.weights]

            if self.activation is not None:
                if issubclass(self.activation, ScalarActivation):
                    post_acts = [self.activation(z) for z in pre_acts]
                elif issubclass(self.activation, VectorActivation):
                    post_acts = [self.activation(pre_acts, index=i) for i in range(len(pre_acts))]
                else:
                    raise ValueError("Undefined Activation function")
                outputs.append(post_acts)
            else:
                outputs.append(pre_acts)
        return np.array(outputs, dtype=object)


class DropOut(Layer):
    weights: np.ndarray

    def __init__(self, rate=0.2):
        self.rate = rate

    def propagate(self, inputs) -> np.ndarray:
        batch, features = inputs.shape
        num_neurons_to_drop = int(features * self.rate)

        outputs = inputs.copy()

        for i in range(batch):
            drop_idx = np.random.choice(features, size=num_neurons_to_drop, replace=False)
            drop_mask = np.ones(features, dtype=bool)
            drop_mask[drop_idx] = False

            outputs[i, drop_idx] = Constant(0)
            # Multiplication with inverse due to unresolved problems with / and Expressions
            outputs[i, drop_mask] = outputs[i, drop_mask] * (1 / (1 - self.rate))

        return outputs