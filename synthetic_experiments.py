"""
Script for running INVASE or NON-INVASE on the synthetic data sets from Yoon et al. (2019)
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from non_invase import NonInvase
from invase import Invase
from synthetic_data import generate_data

# Data constants
N_TRAIN = 10000
N_TEST = 10000
N_FEATURES = 11
N_TOP_SELECTED = 4

# Parser constants
DEFAULT_EPOCHS = 1000
DEFAULT_METHOD = 'NON-INVASE'
DEFAULT_INVASE_COEFF = 0.5
DEFAULT_NON_INVASE_COEFF = 0.001

# Create parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    help="sets variable selection method to use",
    choices=['INVASE', 'NON-INVASE'],
    default=DEFAULT_METHOD
)
parser.add_argument(
    "--data",
    help="sets variable selection method to use",
    choices=['Syn1', 'Syn2', 'Syn3', 'Syn4', 'Syn5', 'Syn6'],
    default='Syn1'
)
parser.add_argument(
    "--epochs",
    help="sets number of epochs to train for",
    type=int,
    default=DEFAULT_EPOCHS
)
args = parser.parse_args()

if args.coeff is None:
    if args.method == 'INVASE':
        args.coeff = DEFAULT_INVASE_COEFF
    else:
        args.coeff = DEFAULT_NON_INVASE_COEFF

# Generate train/test data
x_train, y_train, sel_truth_train = generate_data(args.data, N_TRAIN)
x_test, y_test, sel_truth_test = generate_data(args.data, N_TEST)

# Plot test data
# plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test[:, 0])
# plt.show()

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
    invase_object = Invase(MyModel, SelectorModel, error_fn, norm_coeff=0.5)
elif args.method == 'NON-INVASE':
    invase_object = NonInvase(MyModel, SelectorModel, error_fn, prior_coeff=0.001)
else:
    raise ValueError('invalid method given')

def performance_metrics():
    # Evaluating TPR and FDR for top k selection
    sel_probs = invase_object.selector(x_test)
    selection_pred = np.zeros(N_FEATURES)
    # Get index rank
    index_rank = np.argsort(sel_probs, axis=-1)
    true_positive_rate = np.zeros(N_TEST)
    false_discovery_rate = np.zeros(N_TEST)
    for i in range(N_TEST):
        # Create vector of top k feature selection
        selection_pred[:] = 0
        selection_pred[index_rank[i, -N_TOP_SELECTED:]] = 1
        # Get desired selectin
        selection_truth = sel_truth_test[i]

        # Calculate TPR and FDR
        true_pos = np.dot(selection_truth, selection_pred)
        false_pos = np.dot((1 - selection_truth), selection_pred)
        false_neg = np.dot(selection_truth, (1 - selection_pred))

        true_positive_rate[i] = true_pos / (true_pos + false_neg)
        false_discovery_rate[i] = false_pos / (true_pos + false_pos)

    # Print means and variances
    tpr_mean = np.mean(true_positive_rate)
    tpr_var = np.var(true_positive_rate)
    fdr_mean = np.mean(false_discovery_rate)
    fdr_var = np.var(false_discovery_rate)

    return tpr_mean, tpr_var, fdr_mean, fdr_var

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

print("\nFinal Evaluation:")
tpr_mean, tpr_var, fdr_mean, fdr_var = performance_metrics()
print(f"TPR: mean {tpr_mean}, variance {tpr_var}")
print(f"FDR: mean {fdr_mean}, variance {fdr_var}")

