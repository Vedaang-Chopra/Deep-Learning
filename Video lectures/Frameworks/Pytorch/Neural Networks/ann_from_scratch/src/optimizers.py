import numpy as np

class GradientDescent:
    """
    Base class supporting Batch, Mini-batch, and SGD depending on how data is fed externally.
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layer):
        # TODO (M7 - GD Variants): Implement Batch GD, SGD, Mini-batch update logic
        pass

class MomentumOptimizer:
    def __init__(self, lr=0.01, momentum=0.9, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        # Dictionaries to maintain velocities
        self.v_W = {}
        self.v_b = {}

    def step(self, layer):
        # TODO (M8 - Momentum): Implement Momentum and Nesterov updates
        # Hint: Maintain velocity/moment buffers
        pass

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = {}
        self.v_W = {}
        self.m_b = {}
        self.v_b = {}
        self.t = 0 # timestep

    def step(self, layer):
        # TODO (M9 - Adam): Implement Adam optimizer
        # Hint: bias correction needed in Adam
        pass
