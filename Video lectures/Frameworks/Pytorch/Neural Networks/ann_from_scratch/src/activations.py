import numpy as np

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        x shape: (batch_size, ...)
        Returns: max(0, x)
        """
        self.x = x
        # TODO (M2 - Activation Functions): Implement forward for ReLU
        # Hint: cache x for the backward pass
        pass

    def backward(self, grad_output):
        """
        grad_output shape: (batch_size, ...)
        Returns: gradient with respect to input x
        """
        # TODO (M2 - Activation Functions): Implement backward for ReLU
        # Hint: ReLU backward uses mask where self.x > 0
        pass

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        x shape: (batch_size, ...)
        Returns: 1 / (1 + exp(-x))
        """
        # TODO (M2 - Activation Functions): Implement forward for Sigmoid
        pass

    def backward(self, grad_output):
        """
        grad_output shape: (batch_size, ...)
        """
        # TODO (M2 - Activation Functions): Implement backward for Sigmoid
        # Hint: Sigmoid derivative uses output (out * (1 - out))
        pass

class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        # TODO (M2 - Activation Functions): Implement forward for Tanh
        pass

    def backward(self, grad_output):
        # TODO (M2 - Activation Functions): Implement backward for Tanh
        pass


class Softmax:
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        x shape: (batch_size, num_classes)
        Returns: Softmax probabilities
        """
        # TODO (M2 - Activation Functions): Implement forward for Softmax
        # Hint: Subtract max(x) for numerical stability
        pass

    def backward(self, grad_output):
        # TODO (M2 - Activation Functions): Implement backward for Softmax
        pass


# -----------------------------------------------------------------
# Advanced Activations
# -----------------------------------------------------------------

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.x = None

    def forward(self, x):
        # TODO (M17 - Advanced Activations): Implement forward for LeakyReLU
        pass

    def backward(self, grad_output):
        # TODO (M17 - Advanced Activations): Implement backward for LeakyReLU
        pass

class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.x = None

    def forward(self, x):
        # TODO (M17 - Advanced Activations): Implement forward for ELU
        pass

    def backward(self, grad_output):
        # TODO (M17 - Advanced Activations): Implement backward for ELU
        pass

class GELU:
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        # TODO (M17 - Advanced Activations): Implement forward for GELU
        # Hint: GELU uses Gaussian CDF
        pass
        
    def backward(self, grad_output):
        # TODO (M17 - Advanced Activations): Implement backward for GELU
        pass

def plot_activation_functions():
    """
    Utility to plot activations and their derivatives.
    """
    # TODO (M2 - Activation Functions): Plot activation vs derivative (separate utility function)
    pass
