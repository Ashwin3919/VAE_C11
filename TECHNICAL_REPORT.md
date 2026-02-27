# MNIST CVAE: From Mini to Full

This report documents the evolution of the Conditional Variational Autoencoder (CVAE) from its smallest iteration to the full MNIST model, focusing on architecture scaling, performance throughput, and parallelization benefits.

## Model Scaling

The project transitioned through three distinct configurations, scaling in complexity and capacity to handle increasing numbers of digits.

| Variant | Parameters | Hidden Layers (h1, h2) | Latent Dim | Digits |
| :--- | :--- | :--- | :--- | :--- |
| **Mini** | ~385K | [256, 128] | 32 | 0, 1 |
| **Mid** | ~934K | [512, 256] | 64 | 0, 1 |
| **Full** | ~1.7M | [640, 320] | 128 | 0 – 9 |

The scaling was achieved by increasing the width of the linear layers and the dimensionality of the latent space, allowing the model to capture more complex features required for the full 10-digit classification and generation task.

## Training Performance

Performance was measured in images per second (img/s) on a local CPU. The introduction of OpenMP parallelization significantly improved throughput across all model sizes.

| Variant | Serial (img/s) | OpenMP (img/s) | Speedup |
| :--- | :--- | :--- | :--- |
| **Mini** | ~15,000 | ~45,000 | ~3.0x |
| **Mid** | ~6,000 | ~22,000 | ~3.6x |
| **Full** | ~3,500 | ~14,000 | ~4.0x |

*Note: Benchmarks vary by hardware; these figures represent typical performance on a modern multi-core machine.*

## Parallelization Strategy

Parallelization was implemented using **OpenMP** targeting the most compute-intensive bottleneck: the batched linear transformation (`linear_batch`). By using `#pragma omp parallel for` on the batch dimension, the matrix-vector multiplications for all samples in a mini-batch are computed concurrently. This approach is highly effective for VAEs where the forward and backward passes for each sample in a batch are independent.

## Visual Samples

The `results_main/` directory contains generated PGM images for each model version. These include:
- Reconstructions of input digits.
- Conditional generations (asking for a specific digit).
- Latent space interpolations (watching one digit morph into another).

## Conclusion

This project was a pursuit of understanding generative models at the **system level**. The goal was not to achieve the state-of-the-art in MNIST generation, but to build every component—from the memory slab to the gradient chain—with absolute clarity. The transition from a 385K parameter binary classifier to a 1.7M parameter generative engine highlights the fundamental relationship between model capacity, training efficiency, and low-level system design.