# Technical Report: MNIST Conditional VAE in C

**Author:** Ashwin Shirke
**Language:** C11
**Dependencies:** `libc`, `libm`, optional OpenMP

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
3. [Hyperparameters](#3-hyperparameters)
4. [Memory Model](#4-memory-model)
5. [Forward Pass](#5-forward-pass)
6. [Loss Function (ELBO)](#6-loss-function-elbo)
7. [Backward Pass](#7-backward-pass)
8. [Optimizer](#8-optimizer)
9. [RNG](#9-rng)
10. [Training Loop](#10-training-loop)
11. [GEMM and Performance](#11-gemm-and-performance)
12. [Checkpoint Format](#12-checkpoint-format)
13. [Build System](#13-build-system)
14. [Test Suite](#14-test-suite)


---

## 1. Project Overview

This project is a ground-up implementation of a **Conditional Variational Autoencoder (CVAE)** trained on the MNIST handwritten digit dataset. Every component — the neural network layers, reparameterisation trick, ELBO loss, backpropagation, Adam optimiser, and binary checkpointing — is written from scratch in C11 with no external ML libraries.

**What it does:**
Given a digit label (0–9), the trained model generates new 28×28 pixel images of that digit by sampling from a learned latent space. The model is *conditional*: the class label is one-hot encoded and concatenated into both the encoder input and decoder input, enabling class-controlled generation at inference time.

**Why C:**
Every design decision is forced to be explicit. There is no automatic differentiation, no tensor abstraction, no GPU kernel. The matrix multiplications, gradient accumulators, Adam moment buffers, and reparameterisation trick are all written out and readable. The secondary goal was production-quality engineering: strict module boundaries, a single-slab heap memory model, per-instance RNG with no shared mutable state, a numerical gradient checker, and endian-safe binary checkpoints.

Two trained variants are provided:

| Variant | Digits | Latent dim | Parameters | Epochs | Train time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | 0–1 | 32 | ~385K | 300 | ~30 min |
| **v3** | 0–9 | 64 | ~406K | 400 | ~90 min |

---

## 2. Model Architecture

Both variants are symmetric MLPs (Encoder + Decoder) with ELU activations throughout, and a sigmoid output layer. The hidden layer widths are identical across both variants — only the latent dimension and number of classes differ.

### Encoder

```
enc_in = IMAGE_SIZE + num_classes   →  h1  →  h2  →  μ, log σ²
         (784 + 2 = 786 for v1)         256    128    latent
         (784 + 10 = 794 for v3)
```

The encoder takes the concatenation of the flattened image (784 floats) and a one-hot class label (2 or 10 floats), passes it through two ELU-activated fully-connected layers, then produces two parallel linear projections: **μ (mu)** and **log σ² (logvar)**, each of dimension `latent`.

`logvar` is clamped to `[−4.0, 4.0]` to keep σ in a numerically stable range: `exp(−4/2) ≈ 0.135` to `exp(4/2) ≈ 7.4`.

### Reparameterisation

```
ε ~ N(0, I)          (sampled at training time; ε = 0 at inference)
z = μ + exp(0.5 · logvar) · ε
```

`ε` is stored in `eps_buf` for re-use during the backward pass.

### Decoder

```
dec_in = latent + num_classes   →  h1  →  h2  →  IMAGE_SIZE
         (32 + 2 = 34 for v1)       256    128    784
         (64 + 10 = 74 for v3)
```

The decoder takes the concatenation of the sampled latent vector `z` and the one-hot label, and reconstructs the image through two ELU layers followed by a sigmoid output layer.

### Layer dimensions summary

| Layer | v1 shape | v3 shape | Activation |
| :--- | :--- | :--- | :--- |
| `enc_w1`, `enc_b1` | 786×256, 256 | 794×256, 256 | ELU |
| `enc_w2`, `enc_b2` | 256×128, 128 | 256×128, 128 | ELU |
| `mu_w`, `mu_b` | 128×32, 32 | 128×64, 64 | Linear |
| `lv_w`, `lv_b` | 128×32, 32 | 128×64, 64 | Linear + clamp |
| `dec_w1`, `dec_b1` | 34×256, 256 | 74×256, 256 | ELU |
| `dec_w2`, `dec_b2` | 256×128, 128 | 256×128, 128 | ELU |
| `dec_w3`, `dec_b3` | 128×784, 784 | 128×784, 784 | Sigmoid |

**Why same hidden dims for both variants?** v1 (2 classes) demonstrated that 256/128 is sufficient capacity for MNIST. v3 only needs more latent dimensions — not more hidden width — to separate 10 classes. An earlier v3 design with 640/320 hidden layers (~1.3M params) caused training instability with no quality improvement.

**Weight initialisation:** He initialisation — `w ~ N(0, sqrt(2 / fan_in))` — applied to all weight matrices. The logvar head weights are additionally scaled by 0.01 to keep the initial encoder variance near N(0,1), reducing the risk of early posterior collapse.

---

## 3. Hyperparameters

All hyperparameters are defined in `VAEConfig` (`include/vae_config.h`) and set by the preset constructors in `src/config/vae_config.c`. No values are hardcoded in training logic.

### Architecture parameters

| Parameter | Type | v1 | v3 | Description |
| :--- | :--- | :--- | :--- | :--- |
| `h1` | `int` | 256 | 256 | Width of first hidden layer |
| `h2` | `int` | 128 | 128 | Width of second hidden layer |
| `latent` | `int` | 32 | 64 | Latent space dimensionality |
| `num_classes` | `int` | 2 | 10 | Number of conditioning classes |
| `enc_in` | `int` | 786 | 794 | Encoder input: `IMAGE_SIZE + num_classes` (derived) |
| `dec_in` | `int` | 34 | 74 | Decoder input: `latent + num_classes` (derived) |

### Training parameters

| Parameter | Type | v1 | v3 | Description |
| :--- | :--- | :--- | :--- | :--- |
| `batch_size` | `int` | 64 | 64 | Samples per gradient step |
| `epochs` | `int` | 300 | 400 | Maximum training epochs |
| `lr` | `float` | 0.001 | 0.0001 | Peak learning rate |
| `lr_warmup_epochs` | `int` | 30 | 5 | Linear LR ramp: 0 → `lr` over this many epochs |
| `beta_start` | `float` | 0.001 | 0.001 | Initial KL weight (β) during warmup |
| `beta_end` | `float` | 0.05 | 0.05 | Final KL weight (β) after annealing |
| `beta_warmup` | `int` | 50 | 50 | Epochs before β starts ramping |
| `beta_anneal` | `int` | 100 | 100 | Epochs over which β ramps to `beta_end` |
| `grad_clip` | `float` | 5.0 | 5.0 | Per-element gradient clip magnitude |
| `es_patience` | `int` | 60 | 60 | Early stopping: max epochs without val improvement |
| `es_min_epoch` | `int` | 220 | 220 | Early stopping: never fire before this epoch |
| `save_every` | `int` | 50 | 50 | Checkpoint interval (also triggers sample generation) |

### v3 learning rate rationale

The LR schedule is per-epoch, not per-step. v3 (54K training samples) has approximately 843 batches per epoch versus v1's 178. At the same peak LR:

| | Batches/epoch | Adam steps at warmup end |
| :--- | :--- | :--- |
| v1 | 178 | 5,340 |
| v3 | 843 | 25,290 (4.7× more) |

v3 reaches good reconstruction loss within ~2 epochs (~1,700 Adam steps). Any LR above 0.0001 disrupts a model that has already found a good loss basin. `lr_warmup_epochs=5` was chosen so the LR peaks early and the cosine decay phase covers the full 395 remaining epochs including KL annealing.

### Activation and numeric constants (`include/vae_math.h`)

| Constant | Value | Description |
| :--- | :--- | :--- |
| `IMAGE_SIZE` | 784 | 28×28 MNIST pixels |
| `ELU_ALPHA` | 1.0 | Standard ELU α; prior value of 0.2 gave 5× weaker negative-side gradients |
| `LOGVAR_MIN` | −4.0 | Clamp floor for logvar; prevents near-zero σ that encourages deterministic encoding |
| `LOGVAR_MAX` | 4.0 | Clamp ceiling for logvar |
| `SIGMOID_CLAMP` | 15.0 | Input clamp for sigmoid to avoid exp overflow |
| `ELU_CLAMP` | −15.0 | Input clamp for ELU's `expf()` |
| `GRAD_CLIP_DEFAULT` | 5.0 | Default per-element gradient clip |

---

## 4. Memory Model

Every `VAE` instance allocates a **single contiguous 32-byte-aligned slab** at creation time. All weights, activations, pre-activations, gradient accumulators, Adam moment buffers, and backward scratch buffers are carved out of this slab via pointer arithmetic.

### Two-pass ALLOC_BLOCK macro

A single `ALLOC_BLOCK(M, c, bs)` macro lists every tensor in layout order. It is evaluated twice with different definitions of `NEXT(field, sz)`:

- **Pass 1** (`NEXT_SZ`): `field` is assigned but the value is discarded; `sz` is accumulated into a counter `n`. This computes the total slab size.
- **Pass 2** (`NEXT_PTR`): `field = p; p += sz`. Each pointer is assigned into the allocated slab and the cursor advances.

After Pass 2, a runtime integrity check asserts `p == _mem + n`. This guard uses an explicit `if/abort` rather than `assert()`, which is compiled out by `-DNDEBUG` in release builds.

**Benefits:**
- One `malloc` / one `free` per model
- No heap fragmentation
- 32-byte alignment (= one AVX2 register width) on every float pointer, enabling `vmovaps` (aligned SIMD load) in the GEMM inner loop
- Adding a new tensor requires exactly one line in `ALLOC_BLOCK`

### Slab contents (in order)

1. **Weights** — `enc_w1/b1`, `enc_w2/b2`, `mu_w/b`, `lv_w/b`, `dec_w1/b1`, `dec_w2/b2`, `dec_w3/b3`
2. **Activations** — `enc_h1`, `enc_h2`, `mu`, `logvar`, `z`, `dec_h1`, `dec_h2`, `output` (all `batch_size × dim`)
3. **Input concat buffers** — `enc_in_buf` (`bsz × enc_in`), `dec_in_buf` (`bsz × dec_in`)
4. **Pre-activation buffers** — `pre_eh1`, `pre_eh2`, `pre_dh1`, `pre_dh2`, `pre_out`, `eps_buf`
5. **Backward scratch** — single-sample reuse buffers: `sc_out`, `sc_dh2`, `sc_dh1`, `sc_z`, `sc_mu`, `sc_lv`, `sc_eh2`, `sc_eh1`
6. **Gradient accumulators** — weight-shaped, zeroed each batch by `vae_reset_grads()`
7. **Adam moment buffers** — `m_*` and `v_*` for every weight and bias tensor (initialised to zero, never reset)

---

## 5. Forward Pass

`vae_forward()` in `src/core/vae_forward.c` performs the full encoder-reparameterisation-decoder pass over a batch of `bsz` samples.

### Call sequence

```
1. Build enc_in_buf[bsz, enc_in]:  concat(image[784], one_hot(label, nc))
2. linear_batch → ELU → enc_h1[bsz, h1]
3. linear_batch → ELU → enc_h2[bsz, h2]
4. linear_batch → mu[bsz, latent]
5. linear_batch → logvar[bsz, latent]  (clamped to [LOGVAR_MIN, LOGVAR_MAX])
6. Reparameterisation:
     ε ~ N(0,I)   (or ε = 0 at inference, training=0)
     z = μ + exp(0.5·logvar) · ε
     ε saved to eps_buf for backward
7. Build dec_in_buf[bsz, dec_in]:  concat(z[latent], one_hot(label, nc))
8. linear_batch → ELU → dec_h1[bsz, h1]
9. linear_batch → ELU → dec_h2[bsz, h2]
10. linear_batch → Sigmoid → output[bsz, IMAGE_SIZE]
```

`training=0` sets `ε = 0`, making the forward pass deterministic (uses μ directly). Used for validation loss computation and `vae_decode()`.

### Generation path

`vae_decode()` is a decoder-only forward pass that bypasses the encoder entirely. It calls `linear_single()` instead of `linear_batch()` — a simpler non-tiled, non-OpenMP loop optimised for single-sample throughput.

---

## 6. Loss Function (ELBO)

The training objective is the Evidence Lower Bound (ELBO), minimised as:

```
L = BCE(x, x̂) / IMAGE_SIZE  +  β · KL(q(z|x) ∥ N(0,I)) / latent
```

**Reconstruction loss (BCE per pixel):**
```
BCE(x, x̂) = − Σᵢ [ xᵢ · log(x̂ᵢ) + (1 − xᵢ) · log(1 − x̂ᵢ) ]
```
`x̂ᵢ` is clamped to `[1e-7, 1 − 1e-7]` to prevent `log(0)`. Divided by `IMAGE_SIZE` to normalise to per-pixel scale.

**KL divergence (closed form for diagonal Gaussian vs N(0,I)):**
```
KL = −0.5 · Σⱼ [ 1 + logvar_j − μ_j² − exp(logvar_j) ]
```
Divided by `latent` to normalise to per-dimension scale.

**β (KL weight):** Controls the trade-off between reconstruction fidelity and latent space regularity. Annealed from `beta_start` to `beta_end` during training (see Section 10). Final value `beta_end = 0.05` was tuned to balance per-pixel BCE (~0.15–0.30) and per-dimension KL (~0.5–2.0).

**NaN guard:** Any per-sample term that is non-finite is replaced with `1.0` so a single saturated sample cannot corrupt the batch average.

---

## 7. Backward Pass

`vae_backward()` in `src/core/vae_backward.c` accumulates gradients for one batch by looping over samples. Gradients are accumulated (`+=`) into the slab — `vae_reset_grads()` is the sole zeroing point. The `1/bsz` scale factor is applied at the output layer and propagates naturally through the chain.

### Key gradient derivations (per sample)

| Step | Quantity | Formula |
| :--- | :--- | :--- |
| Output (BCE + sigmoid combined) | `d_out[i]` | `(output_i − x_i) / (IMAGE_SIZE · bsz)` |
| Decoder hidden layers | `d_dh2`, `d_dh1` | Standard linear+ELU backprop |
| Reparameterisation — μ | `d_mu[i]` | `d_z[i] + β · μ_i / (latent · bsz)` |
| Reparameterisation — logvar | `d_lv[i]` | `d_z[i] · 0.5 · σ_i · ε_i + β · 0.5 · (exp(lv_i) − 1) / (latent · bsz)` |
| ELU gate | `d_pre[i]` | `d_h[i] · ELU'(pre_i)` where `ELU'(x) = 1 if x > 0, else α · exp(x)` |
| Encoder hidden layers | `d_eh2`, `d_eh1` | Standard linear+ELU backprop |

`σ_i = exp(0.5 · lv_i)` and `ε_i` are read from `eps_buf` saved during the forward pass.

Under OpenMP, gradient accumulators are written with `#pragma omp atomic` to prevent data races across the batch-parallel loop.

### Numerical gradient check

`tests/test_model.c` verifies the full backprop chain by comparing analytic gradients against finite differences:

```
(L(w + ε) − L(w − ε)) / 2ε    with ε = 1e-4
```

Each weight and bias in all 7 layer matrices is perturbed individually and the relative error is asserted below a threshold. This test runs as part of `make test` and does not require MNIST data.

---

## 8. Optimizer

Adam with standard hyperparameters:

| Hyperparameter | Value |
| :--- | :--- |
| β₁ (first moment decay) | 0.9 |
| β₂ (second moment decay) | 0.999 |
| ε (denominator stabiliser) | 1e-8 |
| Gradient clip | 5.0 (per-element) |

**Update rule:**
```
g̃ = clip(g, −grad_clip, grad_clip)
m_t = β₁ · m_{t-1} + (1 − β₁) · g̃
v_t = β₂ · v_{t-1} + (1 − β₂) · g̃²
m̂_t = m_t / (1 − β₁ᵗ)              (bias correction)
v̂_t = v_t / (1 − β₂ᵗ)
θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)
```

**Design decisions:**
- `adam_update()` is a **stateless helper** — the caller (`VAE`) owns the moment buffers (`m_*`, `v_*`) and step counter (`adam_t`). This keeps the optimiser logic in one place without requiring it to know about the VAE struct layout.
- Gradient clipping is applied **before** moment updates, so extreme individual gradients cannot destabilise the exponential moving averages.
- `adam_update()` does **not** zero gradients. That is solely `vae_reset_grads()`'s responsibility, preventing silent phantom updates from stale accumulations.
- AMSGrad and weight decay are intentionally omitted: the training loss is well-conditioned on MNIST with the existing ELBO formulation, and AMSGrad would require an extra moment buffer (≈ doubling parameter memory) for no measurable gain.

---

## 9. RNG

xorshift-64 for uniform values; Box-Muller transform for standard normals. The spare normal from each Box-Muller pair is cached in `Rng.spare` / `Rng.spare_ready` to avoid discarding half of each pair.

```c
typedef struct {
  uint64_t state;     // xorshift-64 state
  int spare_ready;    // 1 if a cached normal is waiting
  float spare;        // the cached normal value
} Rng;
```

Every `VAE` instance holds its own `Rng` embedded directly in the struct. There is no global or file-scope RNG state. Concurrent training runs or OpenMP threads operating on separate `VAE` instances have zero data races on the RNG.

The xorshift-64 state is initialised so that `state = 0` is replaced with `1` (a zero state produces an all-zero degenerate cycle).

---

## 10. Training Loop

`vae_train.c` implements the full training loop with five interacting schedules.

### Dataset split

After loading MNIST, a Fisher-Yates shuffle (using the model's RNG) is performed once, then the dataset is split 90% train / 10% validation. v1 loads only digits 0–1 (~12K training samples); v3 loads all 10 digits (~54K).

### Per-epoch training

```
For each epoch:
  1. Compute lr (LR schedule)
  2. Compute beta (KL annealing)
  3. Shuffle training partition (Fisher-Yates)
  4. For each mini-batch:
       vae_reset_grads()
       vae_forward(training=1)
       loss = vae_loss(beta)
       if isfinite(loss):
           vae_backward(beta)
           vae_apply_gradients(lr)
       else:
           substitute loss = 1.0, skip backward
  5. Batched validation (training=0, no backward)
  6. Early stopping check
  7. Checkpoint if val_loss is best-so-far
  8. Periodic sample generation (every save_every epochs)
```

### LR schedule

Linear warmup from 0 to `lr` over `lr_warmup_epochs`, then cosine decay to 5% of peak:

```
if epoch < lr_warmup_epochs:
    lr = lr_peak · (epoch + 1) / lr_warmup_epochs

else:
    p = (epoch − lr_warmup_epochs) / (epochs − lr_warmup_epochs)
    lr = lr_peak · 0.5 · (1 + cos(π · p))
    lr = max(lr, lr_peak · 0.05)
```

### KL annealing

β is held at `beta_start` during the warmup phase, then linearly ramped to `beta_end` over `beta_anneal` epochs:

```
if epoch <= beta_warmup:
    beta = beta_start
else:
    p = min((epoch − beta_warmup) / beta_anneal, 1.0)
    beta = beta_start + (beta_end − beta_start) · p
```

For both v1 and v3: warmup ends at epoch 50, annealing finishes at epoch 150. The model is fully training under the regularised ELBO objective from epoch 150 onward.

### Early stopping

Stops if `val_loss` does not improve for `es_patience` epochs, but never before `es_min_epoch`. The constraint `es_min_epoch = 220 > beta_warmup + beta_anneal + buffer (50+100+70)` is critical — see Section 15.

### Non-finite batch handling

If `vae_loss()` returns NaN or Inf for a batch, `vae_backward()` and `vae_apply_gradients()` are both skipped. Calling `apply_gradients` without a preceding backward would apply stale first/second moments, producing a silent phantom parameter update. The batch contributes `1.0` to the epoch loss average instead.

---

## 11. GEMM and Performance

The core matrix multiply is `linear_batch`: `Y[bsz×out] = X[bsz×in] · W[in×out] + b[out]`.

### Tiled GEMM

Iterates over `W` in `GEMM_TILE = 64` column/row blocks. A 64×64 float tile = 16 KB, fitting in a typical 32–64 KB L1 data cache. Each tile of W stays hot across all `bsz` sample rows, reducing W cache-miss rate by O(bsz) versus the naive per-sample scan.

```c
for (i0 = 0; i0 < in_n; i0 += GEMM_TILE)
  for (j0 = 0; j0 < out_n; j0 += GEMM_TILE)
    for (i = i0; i < i_end; i++)
      for (j = j0; j < j_end; j++)
        y[j] += x[i] * W[i * out_n + j]
```

### SIMD vectorisation

All GEMM pointer arguments are declared `restrict`, asserting non-aliasing to the compiler. Combined with `-O3 -march=native -ftree-vectorize`, GCC/Clang auto-vectorises the innermost `j` loop with AVX2 SIMD instructions (SSE/AVX on x86, NEON on ARM). The 32-byte slab alignment ensures `vmovaps` (aligned load) is used instead of the slower `vmovups`.

### OpenMP parallelism

`#pragma omp parallel for schedule(static)` on the `bsz` dimension in `linear_batch`. Each sample row writes to a distinct output slice of `Y` and reads from a distinct row of `X` — zero data races in the forward pass. The backward pass uses `#pragma omp atomic` for gradient accumulation.

OpenMP is compiled out when `_OPENMP` is not defined, so the binary builds without `-fopenmp`.

### Measured throughput (Apple M-series, batch=64)

| Variant | Serial | OpenMP | Speedup |
| :--- | :--- | :--- | :--- |
| v1 | ~1,500 img/s | ~4,100 img/s | ~2.7× |
| v3 | ~1,400 img/s | ~4,100 img/s | ~2.9× |

---

## 12. Checkpoint Format

Little-endian binary, portable across host architectures. Each `float` is written byte-by-byte in canonical LE order, so host endianness is irrelevant.

```
[uint32 LE]  magic   = 0x45415643  ('CVAE' in ASCII)
[uint32 LE]  version = 4
[float* LE]  weights, in layer order:
               enc_w1 enc_b1
               enc_w2 enc_b2
               mu_w   mu_b
               lv_w   lv_b
               dec_w1 dec_b1
               dec_w2 dec_b2
               dec_w3 dec_b3
[uint32 LE]  adam_t  (global step counter)
[float* LE]  Adam moment buffers (m_*, v_*) in same layer order
```

On restart, `load_model()` reads the checkpoint and resumes training from the saved `adam_t` step. The LR and β schedules are epoch-based and recomputed from `epoch` — they are not stored in the checkpoint.

Checkpoint files: `models/vae_v1.bin` (v1), `models/vae_v3.bin` (v3).

---

## 13. Build System

GNU Make with GCC. The `HEADERS` variable lists all public headers as a prerequisite for every binary target — any header change forces a full rebuild, preventing stale-object bugs without per-TU dependency files.

### Compiler flags

```makefile
CFLAGS = -Wall -Wextra -O3 -std=c11 -march=native \
         -ffast-math -funroll-loops -ftree-vectorize \
         -flto -fomit-frame-pointer
```

| Flag | Effect |
| :--- | :--- |
| `-O3` | Full optimisation: inlining, loop transforms, CSE |
| `-march=native` | Target CPU instruction set (enables AVX2/NEON) |
| `-ffast-math` | Relaxes IEEE 754 strictness; enables fused multiply-add |
| `-ftree-vectorize` | Explicit SIMD auto-vectorisation pass |
| `-flto` | Link-time optimisation: inlines across translation units |
| `-fomit-frame-pointer` | Frees a register for the inner loop on x86 |

### Build targets

| Target | Binary | Notes |
| :--- | :--- | :--- |
| `make` / `make mini` | `exe/vae_model` | v1, digits 0–1 |
| `make full` | `exe/vae_model` | v3, all 10 digits (`-DVERSION_V3`) |
| `make omp-mini` | `exe/vae_model_omp_mini` | v1 with OpenMP |
| `make omp-full` | `exe/vae_model_omp_full` | v3 with OpenMP |
| `make omp` | both OMP variants | |
| `make debug` | `exe/vae_model_debug` | `-g -O0 -DDEBUG` |
| `make asan` | `exe/vae_model_asan` | AddressSanitizer + UBSan |
| `make test` | `exe/run_tests` | Compiles and runs all tests |
| `make tsan` | `exe/run_tests_tsan` | ThreadSanitizer — validates OpenMP paths |
| `make clean` | — | Removes `exe/`, `*.o`, `*.pgm`, `results_main/`, `models/` |

### OpenMP on macOS

Apple Clang does not bundle `libomp`. The Makefile uses `brew --prefix libomp` to locate the Homebrew-installed library:

```makefile
LIBOMP_CFLAGS = -Xclang -fopenmp -I$(LIBOMP_PREFIX)/include
LIBOMP_LIBS   = -L$(LIBOMP_PREFIX)/lib -lomp
```

On Linux with GCC, override with: `make omp-mini LIBOMP_CFLAGS=-fopenmp LIBOMP_LIBS=-lgomp`

---

## 14. Test Suite

`make test` compiles and runs `exe/run_tests`. No MNIST data is required. All tests are expected to print `ALL TESTS PASSED`.

| Test file | Coverage |
| :--- | :--- |
| `test_rng.c` | xorshift-64 state transitions, Box-Muller distribution correctness (mean ≈ 0, σ ≈ 1), spare-normal caching, RNG determinism from the same seed |
| `test_optimizer.c` | Adam convergence on a scalar quadratic; gradient ownership contract (apply without backward must not update parameters) |
| `test_model.c` | Forward pass determinism; loss finiteness; checkpoint roundtrip (save then load, compare weights byte-for-byte); **numerical gradient check** across all 7 weight matrices using central finite differences |
| `test_train.c` | Integration: training loss decreases over 5 steps; KL annealing schedule produces correct β values at epochs 0, 50, 100, 150 |

The numerical gradient check in `test_model.c` is the most stringent test. It perturbs each scalar weight by `ε = 1e-4` and compares `(L(w+ε) − L(w−ε)) / 2ε` against the analytic gradient. The test fails if any relative error exceeds the threshold, catching sign errors, missing chain rule terms, or incorrect dimension normalisations.

ThreadSanitizer (`make tsan`) runs the same test binary with `-fsanitize=thread` to verify that OpenMP parallelism in `linear_batch` and `vae_loss` introduces no data races.

---



## References

- Kingma & Welling, [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114), ICLR 2014
- Higgins et al., [*β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*](https://openreview.net/forum?id=Sy2fchgDl), ICLR 2017 — motivation for KL weighting and annealing
- LeCun, Cortes, Burges, [*MNIST handwritten digit database*](http://yann.lecun.com/exdb/mnist/)
- Kingma & Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980), ICLR 2015
