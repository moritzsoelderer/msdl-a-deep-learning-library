## MS Deep Learning Library

This repository contains my personal attempt at creating (a rudimentary) deep learning library including its own autodiff engine. This project is inspired by modern deep learning libraries such as tensorflow and pytorch and is furthermore a way to deepen my theoretical knowledge.

This is in no way meant to be used for professional Deep Learning purposes, as the library is in no way optimized for performance and fully written in Python.
Training neural networks is, however, very well possible even if magnitudes slower than in official implementations. For an example see `src/neural_network_test.py`

This project was inspired by an afternoon of learning for a Deep Learning and AI course at uni and by this video on Youtube of a guy implementing its own Neural Network: https://www.youtube.com/watch?v=cAkMcPfY_Ns. **For the sake of transparency**: No code has been copied from the creator of the mentioned video. This would have, undoubtly, negatively affected the learning experience.

### What is currently implemented?

- Basic Computation Graph/Autodiff Engine
- Linear Layer, DropOut Layer
- Activation Functions (ReLU, Softmax, Sigmoid)
- Losses (Cross-Entropy (with Softmax Activation))
- Optimizers (SGD)
- Basic Neural Network Functionality

### What is not (yet) implemented? (*may be implemented in future*)

- More Advanced Layers (BatchNormalization, etc.)
- CNNs, RNNs
- Persisting Model state
- Activation Functions (e.g. tanh)
- Losses (MSE, RMSE, Huber-Loss etc.)
- Validation set support
- Training History
- EarlyStopping

### What will (most likely) never be implemented? (*too hard*)

- Distributed Training
- Transformer support
- etc.
