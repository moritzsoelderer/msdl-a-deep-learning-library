from typing import Callable
import numpy as np
from sklearn.metrics import accuracy_score

from engine.expression import dot
from engine.array_utils import wrap_into_variables, map_to_value, vec_eval, vec_derive
from engine.activation import Activation, ScalarActivation, Softmax, VectorActivation, ReLU
from engine.loss import Loss
from engine.optimizer import Optimizer


class LinearLayer():
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
                    post_acts = [self.activation(z) for z in pre_acts]  # create new objects
                elif issubclass(self.activation, VectorActivation):
                    post_acts = [self.activation(pre_acts, index=i) for i in range(len(pre_acts))]
                else:
                    raise ValueError("Undefined Activation function")
                outputs.append(post_acts)
            else:
                outputs.append(pre_acts)
        return np.array(outputs, dtype=object)


class NeuralNetwork():
    layers: list[LinearLayer]
    loss: type[Loss]
    optimizer: Optimizer

    def __init__(self, layers: list[LinearLayer], loss: type[Loss], optimizer: Optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def forward_pass(self, X: np.ndarray, y: np.ndarray):
        X = wrap_into_variables(X)
        y = wrap_into_variables(y)
        for layer in self.layers:
            X = layer.propagate(X)
        return X

    def backward_pass(self, loss):
        vec_derive(1)(loss)

    def get_weights(self):
        all_weights = [layer.weights for layer in self.layers]
        return all_weights

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        for i in range(epochs):
            print(f"Epoch: {i}")
            pred = self.forward_pass(X, y)
            vec_eval(pred)
            losses = []
            for logit_row, y_row in zip(pred, y):
                losses.append(self.loss(logit_row, y_row))
            vec_eval(losses)
            y_pred = np.argmax(map_to_value(pred), axis=1)
            y_s = np.argmax(y, axis=1)
            print(accuracy_score(y_s, y_pred))
            self.backward_pass(losses)
            print(([map_to_value(w) for w in self.get_weights()]))
            self.optimizer.step(self.get_weights())
            print(([map_to_value(w) for w in self.get_weights()]))

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pred = self.forward_pass(X, y)
        vec_eval(pred)
        return map_to_value(pred)


