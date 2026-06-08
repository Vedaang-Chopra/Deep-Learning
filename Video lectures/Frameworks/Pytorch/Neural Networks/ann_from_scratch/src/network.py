import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.grad_magnitudes = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x, is_training=True):
        """
        x shape: (batch_size, input_dim)
        """
        # TODO (M1 - Forward Pass): Build forward pipeline
        pass

    def backward(self, grad_output):
        """
        grad_output shape: (batch_size, output_dim)
        """
        # TODO (M4 - Backpropagation): Implement backward propagation across layers
        
        # TODO (M6 - Vanishing Gradients): Add hooks to track gradient magnitudes per layer
        pass

    def update(self, optimizer):
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.W is not None:
                optimizer.step(layer)
