# Guided Learning: Deep Learning End-to-End

Welcome to the **Guided Learning** laboratory environment. This comprehensive suite of exercises is designed to take you from foundational concepts in PyTorch to advanced topics in model training, distributed systems, research extensions, and data-centric rigorous experiments.

## 🎯 Lab Overview

This repository section provides a hands-on, deeply technical progression through the modern deep learning stack. It's structured not just to teach theory, but to build intuitive understanding through explicit, inspectable code. 

**Core Objectives:**
- Understand the mechanics of tensors, autograd, and training loops from first principles.
- Build architectures spanning Computer Vision, NLP (Sequence Models), and generative applications.
- Learn how to scale training efficiently across hardware (memory optimization & distributed training).
- Implement modern research techniques like Contrastive Learning and LoRA.
- Learn the intricacies of deploying and serving models in production.
- Adopt rigorous, data-centric machine learning practises and ablation studies.

---

## 🏗️ Lab Structure

The environment is logically split into 9 core module groups.
Within each group, you will generally find:
*   `notebooks/`: A central, interactive Jupyter Notebook that weaves together theory, code cells, and outputs. This is the primary driver for each lab.
*   `src/`: A collection of Python modules containing the underlying algorithm implementations, architectural blocks, and tools imported by the notebook. 

---

## 📚 Module Progressions

### [Group 1: Tensor & Autograd Insights](./Group_1_Tensor_Autograd)
*   **Notebook:** `01_tensor_autograd_lab.ipynb`
*   **Focus:** Core tensor manipulations, understanding the shape and stride internals, and diagnosing gradient flows. Includes manual implementation of backward passes to demystify PyTorch's `autograd`.
*   **Key Source Files:** `tensor_utils.py`, `grad_diagnostics.py`

### [Group 2: Training Dynamics & Stability](./Group_2_Training_Dynamics)
*   **Notebook:** `02_training_dynamics_stability.ipynb`
*   **Focus:** The mechanics of making neural networks converge successfully. Investigating weight initializations, non-linear activation impacts, selecting and tuning optimizers/learning rate schedules, and establishing absolute reproducibility in experiments.
*   **Key Source Files:** `init_and_activations.py`, `optim_and_schedules.py`, `reproducibility.py`, `stability_tools.py`

### [Group 3: Advanced Vision Systems](./Group_3_Vision_Systems)
*   **Notebook:** `03_vision_systems_lab.ipynb`
*   **Focus:** Constructing modern Computer Vision architectures. From core CNN residual blocks up to semantic segmentation (U-Net) and single-shot object detection (YOLOv1). Includes deep dives into geometric detection metrics (IoU, mAP).
*   **Key Source Files:** `cnn_blocks.py`, `segmentation_unet.py`, `yolo_v1.py`, `detection_metrics.py`

### [Group 4: Sequence Models & Attention](./Group_4_Sequence_Models)
*   **Notebook:** `04_sequence_models_lab.ipynb`
*   **Focus:** Processing temporal and sequential data. Starting with standard RNN paradigms, moving to Sequence-to-Sequence (Seq2Seq), and ultimately diving into Multi-Head Attention, Transformer Blocks, and GPT-style Autoregressive Decoders.
*   **Key Source Files:** `rnn_cells.py`, `seq2seq.py`, `attention.py`, `transformer_blocks.py`, `gpt_decoder.py`

### [Group 5: Efficient Training Systems](./Group_5_Efficient_Training)
*   **Notebook:** `05_efficient_training_systems.ipynb`
*   **Focus:** Scaling models efficiently on single devices. In-depth memory profiling, applying Automatic Mixed Precision (AMP), gradient accumulation to simulate large batches, activation checkpointing, and graph compilation (`torch.compile`).
*   **Key Source Files:** `memory_tools.py`, `amp_and_accum.py`, `checkpointing.py`, `compile_tools.py`, `profiling_tools.py`

### [Group 6: Distributed Training](./Group_6_Distributed_Training)
*   **Notebook:** `06_distributed_training_lab.ipynb`
*   **Focus:** Multi-GPU scaling paradigms. Setting up distributed environments, implementing Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP), synchronizing metrics across nodes, and debugging distributed deadlocks.
*   **Key Source Files:** `dist_setup.py`, `ddp_train.py`, `fsdp_train.py`, `dist_metrics.py`, `dist_debug.py`

### [Group 7: Research Extensions](./Group_7_Research_Extensions)
*   **Notebook:** `07_research_extensions_lab.ipynb`
*   **Focus:** Implementing cutting-edge and custom research techniques without heavily relying on high-level abstractions. Includes custom autograd ops, gradient surgery algorithms, infoNCE/Contrastive losses, Low-Rank Adaptation (LoRA) implemented from scratch, and foundational RL (REINFORCE). 
*   **Key Source Files:** `custom_autograd_ops.py`, `grad_surgery.py`, `contrastive_losses.py`, `lora_from_scratch.py`, `reinforce_toy.py`

### [Group 8: Deployment & Serving](./Group_8_Deployment)
*   **Notebook:** `08_deployment_inference_serving.ipynb`
*   **Focus:** Bridging the gap from research code to production-ready artifacts. Export patterns (TorchScript, ONNX), dynamic inference batching, benchmarking latency/throughput, and creating mocked serving architectures.
*   **Key Source Files:** `export_tools.py`, `batching.py`, `inference_bench.py`

### [Group 11: Data-Centric & Experiment Rigor](./Group_11_Data_Centric)
*   **Notebook:** `11_data_experiment_rigor_lab.ipynb`
*   **Focus:** Adopting a data-first mentality. Establishing dataset manifests and integrity checks, strategic data splitting, comprehensive error analysis methodologies, structured ablation studies, detecting data drift, and thorough experiment logging.
*   **Key Source Files:** `dataset_manifest.py`, `data_checks.py`, `error_analysis.py`, `ablations.py`, `drift_monitoring.py`, `run_logging.py`

---

## 🚀 Getting Started

To dive in, follow these steps:

1.  **Environment Setup**: Ensure your Python environment has PyTorch and standard data science libraries (`numpy`, `matplotlib`, `jupyter`) installed. If using NVIDIA GPUs, ensure CUDA tools are accessible.
2.  **Sequential Path**: If you are new to the codebase, we highly recommend processing sequentially from Group 1 through Group 4 to solidify modeling foundations before tackling systems and optimization labs (Groups 5-8).
3.  **Launch Jupyter**: Navigate to the directory of interest and spin up a Jupyter instance to interact with the notebooks directly.

```bash
cd Group_1_Tensor_Autograd/notebooks
jupyter notebook 01_tensor_autograd_lab.ipynb
```

Happy learning, scaling, and debugging!
