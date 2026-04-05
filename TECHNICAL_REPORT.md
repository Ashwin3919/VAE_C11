# MNIST CVAE: Architecture and Training

This report documents the Conditional Variational Autoencoder (CVAE) design, training strategy, and the reasoning behind key decisions.

## Model Variants

Two configurations are provided. Both use the same hidden layer widths — v3 doubles the latent dimension and opens conditioning to all 10 digit classes.

| Variant | Parameters | h1 / h2 | Latent | Classes | Epochs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | ~385K | 256 / 128 | 32 | 2 (digits 0–1) | 300 |
| **v3** | ~406K | 256 / 128 | 64 | 10 (all digits) | 400 |

v3 uses identical hidden layer widths to v1. The only architectural difference is a doubled latent dimension (32 → 64) to give the model capacity for 5× more classes. An earlier design used larger hidden layers (640/320, ~1.3M params) which caused gradient explosion during LR warmup and was abandoned — the v1 architecture is sufficient for MNIST at any digit count.

---

## KL Annealing and Beta Schedule

The original `beta_end = 0.0005` was the primary cause of poor generation quality. With normalized losses:

```
recon = BCE(x, x̂) / IMAGE_SIZE   ≈ 0.15–0.30  (per pixel average)
kl    = KL(q||p)  / latent        ≈ 0.5–2.0    (per dimension average)
loss  = recon + β · kl
```

At `beta = 0.0005`, the KL contribution (~0.0005) was ~400× weaker than reconstruction (~0.20). The model learned a near-deterministic encoder — posterior `q(z|x)` collapsed to tight clusters far from `N(0,1)`. Sampling `z ~ N(0,1)` at generation time meant sampling outside the decoder's trained distribution: broken outputs.

**Fix:** `beta_end` raised to `0.2`, giving KL and reconstruction comparable weight. Generation quality became clean and consistent.

### Full hyperparameter change log

| Parameter | Old | New (v1) | New (v3) | Reason |
| :--- | :--- | :--- | :--- | :--- |
| `beta_end` | 0.0005 | 0.2 | 0.2 | KL was 400× too weak |
| `beta_warmup` | 100 | 50 | 50 | Reach regularisation sooner |
| `beta_anneal` | 150 | 100 | 100 | β reaches max at epoch 150 |
| `es_min_epoch` | 150 | 220 | 220 | Prevent early stop on pre-KL best val |
| `lr` | 0.001 | 0.001 | 0.0001 | v3 converges in ~2 epochs; higher LR diverges |
| `lr_warmup_epochs` | 30 | 30 | 5 | v3 peaks LR early, cosine decay covers rest |
| `LOGVAR_MIN` | −10.0 | −4.0 | −4.0 | min σ = 0.007 encouraged deterministic encoding |
| `ELU_ALPHA` | 0.2 | 1.0 | 1.0 | Standard ELU; stronger gradients for negative activations |

### Early stopping bug (now fixed)

With the old `es_min_epoch=150`, the best val (0.0932) was recorded around epoch 50 when `beta ≈ 0` (pure reconstruction, not ELBO). The patience counter accumulated from epoch 50 and triggered at epoch 151 — the first epoch after `es_min_epoch` where patience > 60. The saved checkpoint was a near-autoencoder, not a generative model. Setting `es_min_epoch=220` ensures early stopping cannot fire until the model has fully adapted to the regularised latent space.

---

## Learning Rate: Per-Epoch Schedule and Dataset Size

The LR schedule ramps linearly over `lr_warmup_epochs` then applies cosine decay — all measured in epochs, not steps. This creates an asymmetry between v1 and v3:

| | Batches / epoch | Adam steps at epoch 30 |
|---|---|---|
| v1 (12K train samples) | 178 | 5,340 |
| v3 (54K train samples) | 843 | 25,290 |

v3 makes 5× more weight updates per epoch. By epoch 2, v3 has done ~1,700 Adam steps and nearly converged to good reconstruction loss. If LR continues rising past `0.0001`, it disrupts a model that has already found a good basin. Setting `lr=0.0001` with `lr_warmup_epochs=5` caps the peak at a level the converged model can absorb; cosine decay handles the remaining 395 epochs.

---

## Training Performance

Measured on Apple M-series CPU, batch size 64.

| Variant | Serial (img/s) | OpenMP (img/s) | Speedup |
| :--- | :--- | :--- | :--- |
| **v1** | ~1,500 | ~4,100 | ~2.7× |
| **v3** | ~1,400 | ~4,100 | ~2.9× |

Both variants share the same hidden layer widths so throughput is nearly identical. The slight v3 slowdown comes from the larger latent dimension affecting the mu/lv/dec_w1 layers. OpenMP parallelises `linear_batch` over the batch dimension (`#pragma omp parallel for`). Requires `libomp` on macOS (`brew install libomp`).

---

## Parallelisation Strategy

OpenMP targets `linear_batch` — the batched GEMM that dominates runtime. Each sample row writes to a distinct output slice and reads a distinct input row, so the `bsz` loop is data-race free. Gradient accumulation in `vae_backward` uses `#pragma omp atomic` for safe writes into shared weight gradient buffers across threads.

---

## Conclusion

A compact, well-tuned CVAE (~385–406K parameters) is sufficient for high-quality conditional generation on MNIST. The critical finding was that `beta_end` must be set so the KL term is comparable in magnitude to the reconstruction term — not a model capacity problem, but a regularisation problem. Once the latent space is properly regularised, the decoder capacity of the v1 architecture handles clean generation for all 10 digit classes.
