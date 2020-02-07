"""
Script for generating the synthetic data sets from Yoon et al. (2019)
Largely based on code from https://github.com/jsyoon0823/INVASE
"""
import numpy as np

N_FEATURES = 11

def generate_data(data_name, n_examples):
    """Generates synthetic data

    Args:
        data_name: a choice of'Syn1', ..., 'Syn6' - determines the generated dataset
        n_examples: number of examples to generate

    Returns:
        Feature matrix x, response matrix y and ground truth selections
    """
    x = np.random.randn(n_examples, N_FEATURES).astype('float32')

    logit = _get_logit(data_name, x)
    selection_truth = _get_selection_truth(data_name, x)

    prob = np.reshape(np.divide(1, 1 + logit), newshape=(n_examples, 1))
    y = np.random.binomial(1, prob)

    return x, y, selection_truth

def _get_logit(data_name, x):
    """Returns the logit probability for a synthetic dataset"""
    if data_name == 'Syn1':
        logit = np.exp(x[:, 0] * x[:, 1])
    elif data_name == 'Syn2':       
        logit = np.exp(np.sum(np.power(x[:, 2:6], 2), axis=1) - 4.0) 
    elif data_name == 'Syn3':
        logit = np.exp(-10 * np.sin(0.2 * x[:, 6]) + np.abs(x[:, 7]) + x[:, 8] + np.exp(-x[:, 9])  - 2.4)
    else:
        # For complex tasks
        if data_name == 'Syn4':
            logit1 = np.exp(x[:, 0] * x[:, 1])
            logit2 = np.exp(np.sum(np.power(x[:, 2:6], 2), axis=1) - 4.0)
        elif data_name == 'Syn5':
            logit1 = np.exp(x[:, 0] * x[:, 1])
            logit2 = np.exp(-10 * np.sin(0.2 * x[:, 6]) + np.abs(x[:, 7]) + x[:, 8] + np.exp(-x[:, 9])  - 2.4)
        elif data_name == 'Syn6':
            logit1 = np.exp(np.sum(np.power(x[:, 2:6], 2), axis=1) - 4.0)
            logit2 = np.exp(-10 * np.sin(0.2 * x[:, 6]) + np.abs(x[:, 7]) + x[:, 8] + np.exp(-x[:, 9])  - 2.4)
        else:
            raise ValueError('invalid data name given')

        neg_indicator = (x[:, 10] < 0).astype('float32')
        pos_indicator = (x[:, 10] >= 0).astype('float32')
        logit = logit1 * neg_indicator + logit2 * pos_indicator

    return logit

def _get_selection_truth(data_name, x):
    """Returns the desired selection for a synthetic dataset"""
    selection_truth = np.zeros(shape=x.shape)
    if data_name == 'Syn1':
        selection_truth[:, :2] = 1
    elif data_name == 'Syn2':       
        selection_truth[:, 2:6] = 1
    elif data_name == 'Syn3':
        selection_truth[:, 6:10] = 1
    else:
        neg_indices = np.where(x[:, 10] < 0)[0]
        pos_indices = np.where(x[:, 10] >= 0)[0]
        selection_truth[:, 10] = 1
        if data_name == 'Syn4':        
            selection_truth[neg_indices, :2] = 1
            selection_truth[pos_indices, 2:6] = 1
        elif data_name == 'Syn5':        
            selection_truth[neg_indices, :2] = 1
            selection_truth[pos_indices, 6:10] = 1
        elif data_name == 'Syn6':        
            selection_truth[neg_indices, 2:6] = 1
            selection_truth[pos_indices, 6:10] = 1

    return selection_truth
