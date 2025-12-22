# multilayer-perceptron

# Sources
https://github.com/tensorflow/tensorflow

https://github.com/keras-team/keras

https://github.com/keras-team/keras/tree/master/keras/src

## custom_layer.py
Core Dense Layer
https://github.com/keras-team/keras/blob/master/keras/src/layers/core/dense.py

BaseLayer (classe parent)
https://github.com/keras-team/keras/blob/master/keras/src/layers/layer.py

Activations (relu, sigmoid, etc.)
https://github.com/keras-team/keras/blob/master/keras/src/activations/activations.py

### Baselayer - Linear

$$
z = w * X + b
$$

$$
A = f(z)
$$

### Sigmoid
Forward
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


Backward
$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

### ReLU
Forward
$$
\text{ReLU}(z) = \max(0, z)
$$

Backward
$$
\text{ReLU}'(z) =
\begin{cases}
1 & z > 0 \\
0 & z \le 0
\end{cases}
$$

### Softmax

Forward
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$





## custom_model.py
Sequential class
https://github.com/keras-team/keras/blob/master/keras/src/models/sequential.py

Model base class
https://github.com/keras-team/keras/blob/master/keras/src/models/model.py


## optimizers.py
Base Optimizer
https://github.com/keras-team/keras/blob/master/keras/src/optimizers/optimizer.py

SGD
https://github.com/keras-team/keras/blob/master/keras/src/optimizers/sgd.py

Adam
https://github.com/keras-team/keras/blob/master/keras/src/optimizers/adam.py


## losses.py 

Base loss
Binary Crossentropy
CategoricalCrossentropy
https://github.com/keras-team/keras/blob/master/keras/src/losses/loss.py
https://github.com/keras-team/keras/blob/master/keras/src/losses/losses.py

# metrics.py

Base Metric
https://github.com/keras-team/keras/blob/master/keras/src/metrics/metric.py


Accuracy
https://github.com/keras-team/keras/blob/master/keras/src/metrics/accuracy_metrics.py

Precision / Recall / F1
https://github.com/keras-team/keras/blob/master/keras/src/metrics/probabilistic_metrics.py


## callbacks.py
History
https://github.com/keras-team/keras/blob/master/keras/src/callbacks/history.py

EarlyStopping
https://github.com/keras-team/keras/blob/master/keras/src/callbacks/early_stopping.py


## data_processor.py
Data loading (base)
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/data/ops/dataset_ops.py
Preprocessing (normalisation, split, etc.)
https://github.com/keras-team/keras/tree/master/keras/src/layers/preprocessing

## custom_predictor.py
Keras predict()
https://github.com/keras-team/keras/blob/master/keras/src/trainers/trainer.py

## plotting.py


