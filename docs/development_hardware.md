# Development Hardware

This file records the local machine used for experiments in this repository. Runtime and performance results should be interpreted in the context of these specifications.

## Main development machine

| Component | Specification |
|---|---|
| CPU | Intel Core i5-12450H |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM |
| System memory | 16 GB RAM |

## Practical considerations

- NumPy-from-scratch experiments can run on the CPU and may be limited by CPU throughput and available system memory.
- PyTorch experiments can use the NVIDIA GPU when the installed PyTorch build and CUDA environment support it.
- Large datasets, long sequences, large batch sizes, and large model configurations should be chosen with the available RAM and 4 GB of GPU memory in mind.
- Record the actual device, batch size, sequence length, and relevant software versions alongside experiment results. Do not assume every experiment uses the GPU.

## Reproducibility

These specifications describe the primary development machine, not a requirement for using the repository. Individual chapters should document any different hardware used for their reported results.
