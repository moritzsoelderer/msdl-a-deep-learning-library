## MS Deep Learning Library

This repository contains my personal attempt at creating (a rudimentary) deep learning library including its own autodiff engine. This project is inspired by modern deep learning libraries such as tensorflow and pytorch and is furthermore a way to deepen my theoretical knowledge.

This is in no way meant to be used for professional Deep Learning purposes, as the library is in no way optimized for performance and fully written in Python.
Training neural networks is, however, very well possible even if magnitudes slower than in official implementations. For an example see `./showcase.ipynb`

This project was inspired by an afternoon of learning for a Deep Learning and AI course at uni and by this [video on Youtube of a guy implementing its own Neural Network](https://www.youtube.com/watch?v=cAkMcPfY_Ns). **For the sake of transparency**: No code has been copied from the creator of the mentioned video. This would have, undoubtedly, negatively affected the learning experience.

An overview of what IS, MAY BE and WILL NEVER BE implemented can be found below.

### Usage Example

The usage closely aligns what one would expect from tensorflow or a neural network class in python.
```
nn = NeuralNetwork(layers=[
    LinearLayer(4, 32, ReLU),
    LinearLayer(32, 16, ReLU),
    DropOut(rate=0.2),
    LinearLayer(16, 3)
    ],
    loss=CrossEntropyWithLogits,
    optimizer=SGD(learning_rate=0.001, clip_gradients=1.0)
)

X, y = load_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y)

encoder = OneHotEncoder(sparse_output=False)
y_train_cat = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_cat = encoder.fit_transform(y_test.reshape(-1, 1))

nn.train(X_train, y_train_cat, epochs=300, batch_size=24)

predictions = nn.predict(X_test, y_test_cat)
y_pred = np.argmax(predictions, axis=1)
```

### What is currently implemented?

- Basic Computation Graph/Autodiff Engine
- Linear Layer, DropOut Layer
- Activation Functions (ReLU, Softmax, Sigmoid)
- Losses (Cross-Entropy (with Softmax Activation))
- Optimizers (SGD)
- Gradient Clipping
- Basic Neural Network Functionality

### What is not (yet) implemented? (*may be implemented in future*)

- More Advanced Layers (BatchNormalization, etc.)
- CNNs, RNNs
- Persisting Model state
- Activation Functions (e.g. tanh)
- Losses (MSE, RMSE, Huber-Loss etc.)
- Optimizers (Adam, RMSProp)
- Validation set support
- Training History
- EarlyStopping

### What will (most likely) never be implemented? (*too hard and or time-consuming*)

- Distributed Training
- Transformer support
- etc.
