"""
Script for running INVASE or NON-INVASE on the logical AND experiment in the NON-INVASE paper
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from non_invase import NonInvase
from invase import Invase

# Data constants
N_FEATURES = 11
N_TRAIN = 10000
N_TEST = 1000
SCALE = 10

# Parser constants
DEFAULT_EPOCHS = 100
DEFAULT_METHOD = 'NON-INVASE'
DEFAULT_INVASE_COEFF = 0.1
DEFAULT_NON_INVASE_COEFF = 0.01

# Configure environment
np.random.seed(1)
rc('text', usetex=True)

# Create parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    help="sets variable selection method to use",
    choices=['INVASE', 'NON-INVASE'],
    default=DEFAULT_METHOD
)
parser.add_argument(
    "--epochs",
    help="sets number of epochs to train for",
    type=int,
    default=DEFAULT_EPOCHS
)
parser.add_argument(
    "--coeff",
    help="regularisation coefficient",
    type=float,
    default=None
)
args = parser.parse_args()

if args.coeff is None:
    if args.method == 'INVASE':
        args.coeff = DEFAULT_INVASE_COEFF
    else:
        args.coeff = DEFAULT_NON_INVASE_COEFF

# Generate train/test data
def logit(x):
    first = np.divide(1, 1 + np.exp(- SCALE * x[:, 0]))
    second = np.divide(1, 1 + np.exp(- SCALE * x[:, 1]))
    log = (first * second)
    return log

def invase_test(x):
    l = np.exp(x[:, 0] * x[:, 1])

x_train = np.random.normal(0, 1, size=(N_TRAIN, N_FEATURES)).astype('float32')
x_test = np.random.normal(0, 1, size=(N_TEST, N_FEATURES)).astype('float32')

y_train_prob = logit(x_train)
y_train = np.random.binomial(n=1, p=y_train_prob, size=(1, N_TRAIN)).T.astype('float32')

y_test_prob = logit(x_test)
y_test = np.random.binomial(n=1, p=y_test_prob, size=(1, N_TEST)).T.astype('float32')

x_quadrant_test = np.concatenate((
        np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]),
        np.ones((4, 9))
    ), axis=1)

# Plot test data
scatter = plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test[:, 0])
plt.colorbar(scatter)
axes = plt.gca()
axes.set_xlim([-3, 3])
axes.set_ylim([-3, 3])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(r'$y$ over $x_1$ and $x_2$')
plt.show()

# Create train and test datasets
train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Predictor (& baseline) model
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(64, activation='softmax', name='model_dense_1')
        self.d2 = Dense(64, activation='softmax')
        self.d3 = Dense(2, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x

# Selector model
class SelectorModel(Model):
    def __init__(self):
        super(SelectorModel, self).__init__()
        self.d1 = Dense(32, activation='softmax')
        self.d2 = Dense(32, activation='softmax')
        self.d3 = Dense(N_FEATURES, activation='sigmoid')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x

# Error function for training
error_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

# Create INVASE or NON-INVASE loss
if args.method == 'INVASE':
    invase_object = Invase(MyModel, SelectorModel, error_fn, norm_coeff=args.coeff)
elif args.method == 'NON-INVASE':
    invase_object = NonInvase(MyModel, SelectorModel, error_fn, prior_coeff=args.coeff)
else:
    raise ValueError('invalid method given')

# Accuracy metrics
predictor_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='predictor_train_accuracy')
predictor_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='predictor_test_accuracy')

# Train
optimizer = tf.keras.optimizers.Adam()
template = ("Epoch {}:\n"
            "Predictor: train loss {}, test loss {}, train accuracy, test accuracy\n"
            "Selector: train loss {}")

for epoch in range(args.epochs):
    # Reset the metrics at the start of the next epoch
    invase_object.reset_metrics()
    predictor_train_accuracy.reset_states()
    predictor_test_accuracy.reset_states()

    # Train predictor and selector
    for features, labels in train_ds:
        predictions = invase_object.train_step(features, labels, optimizer)
        predictor_test_accuracy(labels, predictions)

    sel_probs_sum = np.zeros(N_FEATURES)
    sel_probs_sqr_sum = np.zeros(N_FEATURES)
    n = 0
    for test_features, test_labels in test_ds:
        predictions = invase_object.test_step(test_features, test_labels)
        predictor_test_accuracy(test_labels, predictions)

        # Get selection probabilities
        output = invase_object.selector(test_features)
        n += output.shape[0]
        sel_probs_sum += np.sum(output, axis=0)
        sel_probs_sqr_sum += np.sum(np.square(output), axis=0)

    sel_probs_mean = sel_probs_sum / N_TEST
    sel_probs_variance = sel_probs_sqr_sum / N_TEST - np.square(sel_probs_mean)

    eval_text = template.format(
        epoch + 1,
        invase_object.predictor_train_loss.result(),
        invase_object.predictor_test_loss.result(),
        predictor_train_accuracy.result() * 100,
        predictor_test_accuracy.result() * 100,
        invase_object.selector_train_loss.result()
    )
    print(eval_text)

    print("Selection probs mean:    ", sel_probs_mean)
    print("Selection probs variance:", sel_probs_variance)

    quadrant_sel_probs = invase_object.selector(x_quadrant_test)
    print(quadrant_sel_probs[:, :2])

# Plot selection prob of first feature on (x_1, x_2) plane
sel_probs = invase_object.selector(x_test)
plt.grid()
scatter = plt.scatter(x_test[:, 0], x_test[:, 1], c=sel_probs[:, 0])
plt.colorbar(scatter)
axes = plt.gca()
axes.set_xlim([-3, 3])
axes.set_ylim([-3, 3])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(r'$\pi^\theta_1(x)$ over $x_1$ and $x_2$')

# Plot selection prob of second feature on (x_1, x_2) plane
plt.figure()
sel_probs = invase_object.selector(x_test)
plt.grid()
scatter = plt.scatter(x_test[:, 0], x_test[:, 1], c=sel_probs[:, 1])
plt.colorbar(scatter)
axes = plt.gca()
axes.set_xlim([-3, 3])
axes.set_ylim([-3, 3])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(r'$\pi^\theta_2(x)$ over $x_1$ and $x_2$')

plt.show()
