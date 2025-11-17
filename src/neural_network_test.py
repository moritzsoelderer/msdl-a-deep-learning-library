import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from engine.loss import CrossEntropyWithLogits
from model.layers import DropOut
from model.neural_network import NeuralNetwork, LinearLayer
from engine.activation import ReLU
from engine.optimizer import SGD


if __name__ == "__main__":
    nn = NeuralNetwork(layers=[
        LinearLayer(4, 12, ReLU),
        LinearLayer(12, 8, ReLU),
        DropOut(rate=0.2),
        LinearLayer(8, 3)
        ],
        loss=CrossEntropyWithLogits,
        optimizer=SGD(learning_rate=0.001)
    )

    X, y = load_iris(return_X_y=True)

    encoder = OneHotEncoder(sparse_output=False)
    y_cat = encoder.fit_transform(y.reshape(-1, 1))

    nn.train(X, y_cat, epochs=300, batch_size=32)
    predictions = nn.predict(X, y_cat)
    
    y_pred = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_true=y, y_pred=y_pred)
    
    print(f"Accuracy: {accuracy}")