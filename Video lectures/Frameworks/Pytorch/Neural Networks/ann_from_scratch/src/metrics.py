import numpy as np

def accuracy_score(y_pred, y_true):
    """
    y_pred: (batch_size, num_classes) or (batch_size,)
    y_true: (batch_size, num_classes) or (batch_size,)
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=-1)
    return np.mean(y_pred == y_true)

def plot_bias_variance_tracking(history):
    """
    Plots training vs validation loss/accuracy to analyze bias/variance over time.
    """
    # TODO (M15 - Bias Variance): Add train vs validation tracking visualizations
    pass

def labels_to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot
