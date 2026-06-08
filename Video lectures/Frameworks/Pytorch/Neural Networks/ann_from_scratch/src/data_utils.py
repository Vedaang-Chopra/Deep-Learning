import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from src.metrics import labels_to_one_hot

def load_mnist_data():
    """
    Fetch MNIST dataset safely.
    """
    print("Fetching MNIST data... this may take a moment.")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.values.astype(np.float32)
    y = mnist.target.values.astype(np.int64)
    return X, y

def preprocess_mnist(X, y, test_size=0.2, random_state=42):
    """
    Normalize features, create train/val split, and one-hot encode.
    """
    # Normalize features to [0, 1] range
    X = X / 255.0
    
    # Flatten if not already (MNIST from openml is already N, 784)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    
    # One-hot encode labels
    y_one_hot = labels_to_one_hot(y, num_classes=10)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_one_hot, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val

def create_batches(X, y, batch_size=64, shuffle=True):
    """
    Generator yielding minibatches.
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
        
    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:min(start_idx + batch_size, num_samples)]
        yield X[batch_indices], y[batch_indices]
