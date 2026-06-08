# Deep Learning From Scratch: Component Practice Project

## Project Goal
This scaffolded project is designed to help you construct a highly modular Artificial Neural Network from the bottom up, using solely NumPy. 
You will piece together 18 individual deep learning modules (from Forward propagate layers to layer norms) by completing the missing math operations.

## Folder Structure
```
ann_from_scratch/
│
├── notebooks/
│   └── mnist_runner.ipynb      <-- Aggregator notebook to run pipeline
│
├── src/
│   ├── activations.py          <-- M2, M17: ReX, Sigmoid, Softmax, GELU
│   ├── losses.py               <-- M3, M13, M18: MSE, CrossEntropy, Focal
│   ├── optimizers.py           <-- M7, M8, M9: SGD, Momentum, Adam
│   ├── layers.py               <-- M1, M4, M5, M12, M16: Dense, Norms, Regularizations
│   ├── network.py              <-- M1, M4, M6: Network sequence loops
│   ├── metrics.py              <-- M15: Accuracy and Bias-Var tracking 
│   ├── data_utils.py           <-- Loading MNIST fully defined
│   ├── train_utils.py          <-- M10, M11, M14: Schedulers, dropout, train loop
│   └── config.py               <-- Global constants
│
├── requirements.txt
└── README.md
```

## Implementation Roadmap
DO NOT dive into this blindly! Follow this exact ordering so the dependency chain is respected.

1. **M1**: Forward Pass (`layers.py`, `network.py`)
2. **M2**: Activations (`activations.py`)
3. **M3**: Loss Functions (`losses.py`)
4. **M4**: Backpropagation (`layers.py`, `network.py`) 
5. **M5**: Weight Initialization (`layers.py`)
6. **M7**: Optimizers & SGD Variants (`optimizers.py`, `train_utils.py`)
7. **M9**: Adam Optimizer (`optimizers.py`)
8. **M10**: Learning Rate Scheduling (`train_utils.py`)
9. **M12**: Regularization (`layers.py`)
10. **M14**: Dropout (`train_utils.py`)
11. **M16**: Normalization (`layers.py`)

Follow up with edge modules (M6 Gradient Hooks, M8 Momentum, M13 Analytical Probability, M15 Bias tracking, M17 Advanced Activations, M18 Focal Limits).

## Module Checklist
Use this to track your completion:
- [ ] M1: Complete Forward Pass operations
- [ ] M2: Activation Functions mapped
- [ ] M3: Loss Function errors calculated
- [ ] M4: Backpropagation logic finalized
- [ ] M5: Weight Initialization matrices built
- [ ] M6: Vanishing Gradients trackers placed
- [ ] M7: GD Variants and Batch Loops implemented
- [ ] M8: Momentum running averages
- [ ] M9: Adam Bias Corrections integrated
- [ ] M10: LR Scheduling mapped
- [ ] M11: Loss Landscape stub plotted
- [ ] M12: Regularization boundaries coded
- [ ] M13: Probabilistic Entropies mapped
- [ ] M14: Inverted Dropout applied
- [ ] M15: Bias Variance tracked
- [ ] M16: Batch/Layer Normalizations functioning
- [ ] M17: Advanced Activations
- [ ] M18: Specialized Losses integrated
