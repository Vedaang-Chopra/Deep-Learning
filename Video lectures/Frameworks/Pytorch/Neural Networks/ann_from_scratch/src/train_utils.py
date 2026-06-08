import numpy as np
import matplotlib.pyplot as plt
from src.data_utils import create_batches
from src.metrics import accuracy_score

# -----------------------------------------------------------------
# Modules: Dropout & Schedulers
# -----------------------------------------------------------------
class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, is_training=True):
        # TODO (M14 - Dropout): Implement inverted dropout
        pass

    def backward(self, grad_output):
        # TODO (M14 - Dropout): Implement inverted dropout backward
        pass

class LRScheduler:
    def __init__(self, initial_lr, total_epochs, warmup_epochs=0):
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
    def get_lr(self, epoch):
        # TODO (M10 - LR Scheduling): Add learning rate scheduler (cosine / warmup)
        pass


# -----------------------------------------------------------------
# Training Loops
# -----------------------------------------------------------------

def train_one_epoch(model, loss_fn, optimizer, X_train, y_train, batch_size):
    """
    Returns average loss and accuracy.
    """
    # TODO (M7 - GD Variants): Implement training loop
    pass

def evaluate(model, loss_fn, X_val, y_val, batch_size=64):
    """
    Returns average validation loss and accuracy.
    """
    # TODO (M7 - GD Variants): Implement evaluation loop
    pass

def fit(model, loss_fn, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    """
    Full training logic.
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # TODO (M11 - Loss Landscape): Add hooks to track loss over time
    
    for epoch in range(epochs):
        # TODO (M10 - LR Scheduling): Update optimizer LR dynamically if scheduler is provided
        
        train_loss, train_acc = train_one_epoch(model, loss_fn, optimizer, X_train, y_train, batch_size)
        val_loss, val_acc = evaluate(model, loss_fn, X_val, y_val, batch_size)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
              
    return history
