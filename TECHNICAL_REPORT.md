# Technical Report: MNIST CVAE in C

This document covers the implementation decisions, training dynamics, and engineering details behind the CVAE. For general usage see the README.

---

## Model Architecture

Two variants are provided, sharing the same hidden layer widths:

| Variant | h1 / h2 | Latent | Classes | Params | Epochs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | 256 / 128 | 32 | 2 (digits 0–1) | ~385K | 300 |
| **v3** | 256 / 128 | 64 | 10 (all digits) | ~406K | 400 |

The encoder and decoder are symmetric MLPs with ELU activations. The label is one-hot encoded and concatenated at both encoder input and decoder input:

```
enc_in = IMAGE_SIZE + num_classes   (784 + 2 or 784 + 10)
dec_in = latent     + num_classes   (32+2  or  64+10)
```

**Why the same hidden dims for both?** v1 (2 classes) proved 256/128 is sufficient capacity for MNIST. v3 only needs more latent dimensions — not more hidden width — to separate 10 classes. An earlier v3 design with 640/320 hidden layers (~1.3M params) caused training instability with no quality benefit.

---

## Memory Model

Every `VAE` instance allocates a **single contiguous 32-byte-aligned slab** at creation time. All weights, activations, gradient accumulators, and Adam moment buffers are carved out of this slab via pointer arithmetic using a two-pass macro (`ALLOC_BLOCK`):

- **Pass 1** (`NEXT_SZ`): accumulates total byte count
- **Pass 2** (`NEXT_PTR`): assigns each pointer into the allocated slab

After both passes, a runtime integrity check verifies `p == _mem + slab_size()`. This guard is always active (not `assert`, which `-DNDEBUG` removes) and prints a byte-level delta on mismatch.

Benefits: one allocation, one free, no fragmentation, guaranteed 32-byte alignment for AVX2 SIMD loads throughout the GEMM inner loop.

---

## GEMM and Performance

The core operation is `linear_batch`: `Y[bsz×out] = X[bsz×in] * W[in×out] + b[out]`.

**Tiled GEMM** — iterates in `GEMM_TILE=64` blocks. A 64×64 float tile = 16 KB, fitting in a typical 32–64 KB L1 data cache. Each tile of W stays hot while all `bsz` sample rows are accumulated against it, reducing W cache-miss rate by O(bsz) versus the naive per-sample scan.

**`restrict` pointers** — all arguments declared `restrict`, asserting non-aliasing to the compiler. Combined with `-O3 -march=native -ftree-vectorize`, this enables SIMD auto-vectorisation of the inner dot-product loop.

**OpenMP** — `#pragma omp parallel for` on the `bsz` dimension. Each sample row writes to a distinct output slice, so there are zero data races in the forward pass. Backward pass uses `#pragma omp atomic` for gradient accumulation.

**Measured throughput on Apple M-series CPU, batch=64:**

| Variant | Serial | OpenMP | Speedup |
| :--- | :--- | :--- | :--- |
| v1 | ~1,500 img/s | ~4,100 img/s | ~2.7× |
| v3 | ~1,400 img/s | ~4,100 img/s | ~2.9× |

---

## Training Design

### LR Schedule
Linear warmup for `lr_warmup_epochs` epochs, then cosine decay to 5% of peak.

### KL Annealing
β held at `beta_start` (~0) during warmup, linearly ramped to `beta_end=0.2` over `beta_anneal` epochs. This gives the reconstruction loss time to converge before the latent space is regularised.

### Early Stopping
Stops if validation loss does not improve for `es_patience` epochs, but never before `es_min_epoch`. The `es_min_epoch` constraint is critical — see the bug below.

### Checkpointing
Best validation loss checkpoint is saved every `save_every` epochs and reloaded automatically on restart. Training is fully resumable.

---

## Key Findings and Bugs Fixed

### 1. Posterior Collapse — `beta_end` was 400× too small

The original `beta_end = 0.0005` caused completely broken generation. With normalized losses:

```
recon = BCE / IMAGE_SIZE  ≈ 0.15–0.30
kl    = KL  / latent      ≈ 0.5–2.0
```

At `beta=0.0005`, KL contribution ≈ 0.0005 — roughly **400× weaker** than reconstruction. The model learned a near-deterministic encoder: `q(z|x)` collapsed to tight clusters far from `N(0,1)`. Sampling `z ~ N(0,1)` at generation time meant sampling outside the decoder's trained distribution — broken outputs.

**Fix:** `beta_end = 0.2`. KL and reconstruction are now comparably weighted. Generation quality became clean and consistent.

### 2. Early Stopping Triggered on the Wrong Checkpoint

With `es_min_epoch=150`, the best validation loss (0.0932) was recorded around epoch 50 when `beta ≈ 0` — essentially pure reconstruction loss, not ELBO. When beta started ramping at epoch 50, validation loss naturally rose (it now includes KL). The patience counter had been accumulating since epoch 50 and triggered at epoch 151 — the first epoch after `es_min_epoch` where patience exceeded the limit. The saved model was a near-autoencoder, not a generative model.

**Fix:** `es_min_epoch = 220` (> `beta_warmup + beta_anneal + buffer` = 50+100+70). Early stopping cannot fire until the model has fully adapted to the regularised latent space.

### 3. v3 LR — Per-Epoch Schedule vs Dataset Size

The LR schedule is per-epoch, not per-step. v3 (54K training samples) has 843 batches/epoch vs v1's 178. At the same peak LR and warmup duration:

| | Batches/epoch | Adam steps at warmup end |
|---|---|---|
| v1 | 178 | 5,340 |
| v3 | 843 | 25,290 (5× more) |

v3 converges to good reconstruction loss in ~2 epochs (~1,700 steps). Any LR above 0.0001 disrupts a model that has already found a good basin.

**Fix:** `lr=0.0001`, `lr_warmup_epochs=5` for v3. LR peaks early; cosine decay covers the remaining 395 epochs including the full KL annealing phase.

### 4. Activation and Logvar Constants

| Constant | Old | New | Reason |
|---|---|---|---|
| `ELU_ALPHA` | 0.2 | 1.0 | Standard ELU; 0.2 gave 5× weaker gradients for negative pre-activations |
| `LOGVAR_MIN` | −10.0 | −4.0 | Min σ = exp(−5) ≈ 0.007 at −10; near-zero variance encouraged deterministic encoding |

---

## Backward Pass

`vae_backward()` loops over samples and accumulates gradients into the slab. All gradients are accumulated (`+=`), not overwritten — `vae_reset_grads()` is the sole point of zeroing. Scale by `1/bsz` is applied at the output layer and propagates through the chain.

Key derivations (per sample):

| Step | Formula |
|---|---|
| Output (BCE+sigmoid combined) | `d_out[i] = (output_i − x_i) / (IMAGE_SIZE · bsz)` |
| Reparameterisation — mu | `d_mu[i] = d_z[i] + β·mu_i / (latent·bsz)` |
| Reparameterisation — logvar | `d_lv[i] = d_z[i]·0.5·σ·ε + β·0.5·(exp(lv_i)−1) / (latent·bsz)` |
| ELU gate | `d_pre[i] = d_h[i] · ELU'(pre_i)` where `ELU'(x) = 1 if x>0 else α·exp(x)` |

A **numerical gradient check** in `test_model.c` verifies the full chain rule by comparing analytic gradients against finite differences: `(L(w+ε) − L(w−ε)) / 2ε`.

---

## Checkpoint Format

Little-endian binary, portable across architectures:

```
[uint32 LE]  magic   = 0x45415643  ('CVAE')
[uint32 LE]  version = 4
[float*  LE] weights: enc_w1 enc_b1 enc_w2 enc_b2 mu_w mu_b lv_w lv_b
                      dec_w1 dec_b1 dec_w2 dec_b2 dec_w3 dec_b3
[uint32 LE]  adam_t
[float*  LE] Adam m and v buffers (same layer order)
```

Each float is written byte-by-byte in canonical LE order — host endianness is irrelevant.

---

## RNG

xorshift-64 for uniform values, Box-Muller transform for normals. Spare normal cached to avoid wasting half of each Box-Muller pair. State is owned per `VAE` instance — no shared mutable state, no data races under OpenMP.

---

## Optimizer

Adam with `β₁=0.9`, `β₂=0.999`, `ε=1e-8`. Implemented as a stateless helper: the caller owns the moment buffers and step counter. Per-element gradient clipping applied before moment updates. `adam_update()` does **not** zero gradients — that is solely `vae_reset_grads()`'s responsibility, preventing silent phantom updates from stale moments.
