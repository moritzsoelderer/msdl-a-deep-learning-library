from enum import Enum
from typing import Callable
import numpy as np
from sklearn.metrics import accuracy_score

from engine.array_utils import wrap_into_variables, map_to_value, vec_eval, vec_derive
from engine.loss import Loss
from engine.optimizer import Optimizer

from model.layers import DropOut, Layer

class PropagationMode(Enum):
    TRAINING = 1
    PREDICTION = 2


class NeuralNetwork():
    layers: list[Layer]
    loss: type[Loss]
    optimizer: Optimizer

    def __init__(self, layers: list[Layer], loss: type[Loss], optimizer: Optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def forward_pass(self, X: np.ndarray, y: np.ndarray, mode: PropagationMode):
        X = wrap_into_variables(X)
        y = wrap_into_variables(y)
        for layer in self.layers:
            if mode == PropagationMode.PREDICTION and isinstance(layer, DropOut):
                continue
            X = layer.propagate(X)
        vec_eval(X)
        return X

    def backward_pass(self, loss):
        vec_derive(1)(loss)

    def get_weights(self):
        all_weights = [layer.weights for layer in self.layers if not isinstance(layer, DropOut)]
        return all_weights

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        for i in range(epochs):
            print(f"Epoch: {i}")
            for j in range(int(X.shape[0] / batch_size)):
                batch = X[j * batch_size:(j+1)*batch_size]
                batch_y = y[j * batch_size:(j+1)*batch_size]

                pred = self.forward_pass(batch, batch_y, PropagationMode.TRAINING)
                losses = self.compute_loss(pred, batch_y)
                self.backward_pass(losses)
                self.optimizer.step(self.get_weights())

                y_pred = np.argmax(map_to_value(pred), axis=1)
                y_s = np.argmax(batch_y, axis=1)
                print(f"Batch {j} - Accuracy: {accuracy_score(y_s, y_pred)}")

    def compute_loss(self, X, y):
        losses = []
        for logit_row, y_row in zip(X, y):
            losses.append(self.loss(logit_row, y_row))
        vec_eval(losses)
        return losses

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pred = self.forward_pass(X, y, PropagationMode.PREDICTION)
        return map_to_value(pred)


