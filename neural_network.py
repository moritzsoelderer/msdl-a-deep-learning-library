from typing import Callable
import numpy as np

from numpy._core.multiarray import scalar

from expressions import Activation, Sigmoid, Softmax, wrap_into_variables, dot, map_to_value, vec_eval, vec_derive, ReLU
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

class LinearLayer():
    weights: np.ndarray
    activation: type[Activation]
    activation_mode: str

    def __init__(self, num_inputs: int, num_neurons: int, activation: type[Activation]) -> None:
        self.weights = wrap_into_variables(np.random.random_sample(size=(num_neurons, num_inputs)))
        self.activation = activation

    def propagate(self, inputs: np.ndarray) -> np.ndarray:
        outputs = []
        for sample in inputs:
            pre_acts = [dot(sample, neuron_weights) for neuron_weights in self.weights]

            if self.activation.mode == "scalar":
                post_acts = [self.activation(z) for z in pre_acts]  # create new objects
            else:
                post_acts = [self.activation(pre_acts, index=i) for i in range(len(pre_acts))]

            outputs.append(post_acts)
        return np.array(outputs, dtype=object)


class NeuralNetwork():
    layers: list[LinearLayer]
    loss: Callable
    optimizer: Callable

    def __init__(self, layers: list[LinearLayer], loss: Callable, optimizer: Callable):
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

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        for i in range(100):
            print(f"Iteration: {i}, ({X.shape}, {y.shape})")
            pred = self.forward_pass(X, y)
            loss = self.loss(pred, y)
            vec_eval(loss)
            self.backward_pass(loss)

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pred = self.forward_pass(X, y)
        vec_eval(pred)
        return map_to_value(pred)


if __name__ == "__main__":
    nn = NeuralNetwork(layers=[
        LinearLayer(4, 10, ReLU),
        LinearLayer(10, 3, Softmax)
        ],
        loss= lambda y_pred, y_true: y_pred - y_true,
        optimizer= lambda theta, delta: theta - 0.001 * delta 
    )
    X, y = load_iris(return_X_y=True)
    encoder = OneHotEncoder(sparse_output=False)
    y_cat = encoder.fit_transform(y.reshape(-1, 1))
    nn.train(X, y_cat)
    predictions = nn.predict(X, y_cat)
    y_pred = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_true=y, y_pred=y_pred)
    
    print(f"Accuracy: {accuracy}")