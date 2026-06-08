import numpy as np

class Dense:
    def __init__(self, in_dim, out_dim, init_type="random"):
        """
        in_dim: size of input feature vector
        out_dim: size of output feature vector
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # TODO (M5 - Weight Initialization): Implement random init, Xavier, and He
        self.W = None  # shape: (in_dim, out_dim)
        self.b = None  # shape: (1, out_dim)
        
        # Placeholders for gradients
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        """
        x shape: (batch_size, in_dim)
        """
        self.x = x
        # TODO (M1 - Forward Pass): Implement Dense forward pass
        # Hint: cache input for backward
        pass

    def backward(self, grad_output):
        """
        grad_output shape: (batch_size, out_dim)
        Returns dX shape: (batch_size, in_dim)
        """
        # TODO (M4 - Backpropagation): Implement gradient computation for this layer
        # Implement backward pass: dW, db, dX
        # Hint: upstream gradient × local gradient
        # Hint: gradients must match parameter shapes
        pass

# -----------------------------------------------------------------
# Normalization Layers
# -----------------------------------------------------------------
class BatchNorm:
    def __init__(self, dim, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))
        
    def forward(self, x, is_training=True):
        # TODO (M16 - Normalization): Implement BatchNorm forward
        # Hint: track running mean/variance
        # Hint: different behavior train vs eval
        pass
        
    def backward(self, grad_output):
        # TODO (M16 - Normalization): Implement BatchNorm backward
        pass

class LayerNorm:
    def __init__(self, dim, epsilon=1e-5):
        self.epsilon = epsilon
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        
    def forward(self, x):
        # TODO (M16 - Normalization): Implement LayerNorm forward
        pass
        
    def backward(self, grad_output):
        # TODO (M16 - Normalization): Implement LayerNorm backward
        pass

# -----------------------------------------------------------------
# Regularization (optional inclusion)
# -----------------------------------------------------------------
class L1Regularization:
    def __init__(self, lambda_l1):
        self.lambda_l1 = lambda_l1
        
    def forward(self, weights):
        # TODO (M12 - Regularization): Implement L1 penalty
        pass
        
    def backward(self, weights):
        # TODO (M12 - Regularization): Implement L1 gradient
        pass

class L2Regularization:
    def __init__(self, lambda_l2):
        self.lambda_l2 = lambda_l2
        
    def forward(self, weights):
        # TODO (M12 - Regularization): Implement L2 penalty
        pass
        
    def backward(self, weights):
        # TODO (M12 - Regularization): Implement L2 gradient
        pass

class ElasticNetRegularization:
    def __init__(self, lambda_l1, lambda_l2):
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        
    def forward(self, weights):
        # TODO (M12 - Regularization): Implement ElasticNet penalty
        pass
        
    def backward(self, weights):
        # TODO (M12 - Regularization): Implement ElasticNet gradient
        pass
