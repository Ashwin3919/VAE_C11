# MNIST Conditional VAE in C

A ground-up implementation of a **Conditional Variational Autoencoder (CVAE)** in pure C11, trained on MNIST, with zero external dependencies beyond `libc` and `libm`.

**Author:** : Ashwin Shirke


---

## Motivation

This project is a deliberate exercise in understanding a machine learning model at its lowest level.

The starting point: *can I implement a VAE — the math, the training loop, the backpropagation — without hiding anything behind a library?*

Writing it in C means every design choice is visible and intentional. There is no automatic differentiation, no tensor abstraction, no GPU kernel. The matrix multiplies, the gradient accumulators, the Adam moment buffers, the reparameterisation trick — all written explicitly, in the open, readable in a single afternoon.

The secondary goal was to write that code at a standard that would hold up under engineering review: strict module boundaries, a single-slab heap memory model, per-instance RNG with no shared mutable state, a numerical gradient check in the test suite, and endian-safe binary checkpoints. This is not about C being the right tool for ML in production. It is about using C as a substrate that forces clarity.

---

## Quick Start

```bash
# 1. Get the MNIST data (one-time download)
./download_mnist.sh

# 2. Build and train the mini model (digits 0 & 1, ~385K params)
make
./exe/vae_model

# 3. Train the full model (all 10 digits, ~1.7M params)
make full
./exe/vae_model

# 4. View generated PGM images
./convert_to_png.sh          # converts results_main/*/  to PNG

# 5. Run the test suite (no MNIST data required)
make test

# 6. Run with AddressSanitizer + UBSan (development)
make asan
./exe/vae_model_asan
```

Checkpoints are saved to `models/vae_vN.bin` every `save_every` epochs and reloaded automatically on restart — training is fully resumable.

---

## Build Targets

| Target | Binary | Model | Digits | Params | Epochs |
|---|---|---|---|---|---|
| `make` / `make all` | `exe/vae_model` | v1 Mini | 0 & 1 | ~385K | 300 |
| `make mid` / `v2` | `exe/vae_model` | v2 Mid | 0 & 1 | ~934K | 400 |
| `make full` | `exe/vae_model` | v3 Full | 0 – 9 | ~1.3M | 800 |
| `make omp-mini` | `exe/vae_model_omp_mini` | v1 + OMP | 0 & 1 | ~385K | — |
| `make omp-mid` | `exe/vae_model_omp_mid` | v2 + OMP | 0 & 1 | ~934K | — |
| `make omp-full` | `exe/vae_model_omp_full` | v3 + OMP | 0 – 9 | ~1.3M | — |
| `make omp` | all three above | — | — | — | — |
| `make debug` | `exe/vae_model_debug` | v1 | 0 & 1 | ~385K | — |
| `make asan` | `exe/vae_model_asan` | v1 | 0 & 1 | ~385K | — |
| `make test` | `exe/run_tests` | — | — | — | — |
| `make tsan` | `exe/run_tests_tsan` | — | — | — | — |
| `make clean` | — | — | — | — | — |

The `omp-mini/mid/full` targets build with `-Xclang -fopenmp` (Apple Clang) or `-fopenmp` (GCC on Linux), activating `#pragma omp parallel for` inside `linear_batch`. All three variants exist so you can compare per-model throughput improvement directly. Requires `libomp` — on macOS: `brew install libomp`.
The `debug` target disables optimisation and enables `-g` for clean stack traces.
The `asan` target builds with `-fsanitize=address,undefined` for catching memory errors and undefined behaviour.
The `tsan` target builds with `-fsanitize=thread` to validate OpenMP paths for data races.

---

## Codebase Layout

```
.
├── src/
│   ├── main.c                    # Entry point — selects config, trains, generates
│   ├── config/vae_config.c       # Runtime config presets (v1 / v2 / v3)
│   ├── core/vae_model.c          # Forward pass, ELBO loss, backprop, Adam step
│   ├── optimizer/vae_optimizer.c # Stateless adam_update() helper
│   ├── rng/vae_rng.c             # xorshift-64 + Box-Muller, per-instance state
│   ├── train/vae_train.c         # Training loop: LR schedule, KL annealing,
│   │                             #   batched validation, early stopping
│   ├── generate/vae_generate.c   # Conditional and interpolated sample generation
│   ├── io/vae_io.c               # Checkpoint save / load (explicit LE binary format)
│   └── mnist_loader.c            # IDX file parser, digit-class filtering
│
├── include/                      # Public headers — one per module
│   ├── vae_config.h
│   ├── vae_math.h                # Inline activations, named constants
│   ├── vae_model.h
│   ├── vae_optimizer.h
│   ├── vae_rng.h
│   ├── vae_train.h
│   ├── vae_io.h
│   ├── vae_generate.h
│   └── mnist_loader.h
│
└── tests/
    ├── test_framework.h          # Zero-dependency ASSERT_EQ / ASSERT_NEAR / RUN_TEST
    ├── test_rng.c                # RNG determinism, uniform/normal distribution stats
    ├── test_optimizer.c          # Adam convergence, gradient-ownership contract
    ├── test_model.c              # Determinism, loss, checkpoint roundtrip,
    │                             #   backward gradients, numerical gradient check
    ├── test_train.c              # Integration: loss decreases over 3 epochs
    └── run_tests.c               # Entry point — accumulates suite totals
```

Each module exposes a single header and has no knowledge of others beyond what it explicitly includes. Changes to the optimizer do not recompile the RNG.

---

## Architecture

This is a **Conditional VAE (CVAE)**. The digit label is one-hot encoded and concatenated directly into both the encoder input and the decoder input, allowing the model to generate a specific digit on demand.

```
                ┌──────────────────── ENCODER ──────────────────────┐
                │                                                    │
 image (784) ─→ │  Linear(enc_in→h1, ELU)  →  Linear(h1→h2, ELU)  │ ─→ μ  [latent]
 label (nc)  ─→ │                              Linear(h2→latent)    │ ─→ log σ²  [latent]
                └────────────────────────────────────────────────────┘

                      ┌─── REPARAMETERISATION ───┐
                      │  z = μ + σ · ε,  ε ~ N(0,I)  │
                      └──────────────────────────────┘

                ┌──────────────────── DECODER ──────────────────────┐
                │                                                    │
 z (latent)  ─→ │  Linear(dec_in→h1, ELU)  →  Linear(h1→h2, ELU)  │ ─→ x̂  [784]
 label (nc)  ─→ │                              Linear(h2→784, Sigmoid) │
                └────────────────────────────────────────────────────┘
```

**enc_in** = IMAGE_SIZE + num_classes
**dec_in** = latent + num_classes

**Loss (ELBO)**:
```
L = BCE(x, x̂) / IMAGE_SIZE  +  β · KL(N(μ,σ²) ∥ N(0,I)) / latent
```

`β` is linearly annealed from `beta_start` → `beta_end` over `beta_anneal` epochs after a `beta_warmup` phase, giving reconstruction space to converge before regularisation is tightened.

---

## Configuration

All hyperparameters and paths live in `VAEConfig`. There are no compile-time `#define` knobs for architecture sizes. Three presets are provided; any field can be overridden after calling a preset constructor:

```c
VAEConfig cfg = vae_config_v1();  // start from a preset
cfg.latent           = 48;        // override whatever you need
cfg.epochs           = 500;
cfg.lr_warmup_epochs = 50;
cfg.data_dir         = "/datasets/mnist";
VAE *model = create_vae(&cfg, /*rng_seed=*/42ULL);
if (!model) { /* handle OOM */ }
```

### Full field reference

| Field | Type | Description |
|---|---|---|
| `h1`, `h2` | `int` | Hidden layer widths (encoder and decoder share the same widths) |
| `latent` | `int` | Latent vector dimension |
| `num_classes` | `int` | Number of conditioning classes (2 for binary, 10 for full MNIST) |
| `enc_in` | `int` | Derived: `IMAGE_SIZE + num_classes` — do not set manually |
| `dec_in` | `int` | Derived: `latent + num_classes` — do not set manually |
| `batch_size` | `int` | Mini-batch size |
| `epochs` | `int` | Total training epochs |
| `lr` | `float` | Peak learning rate (Adam) |
| `beta_start` | `float` | KL weight at the start of annealing |
| `beta_end` | `float` | KL weight at the end of annealing |
| `grad_clip` | `float` | Per-element gradient clip threshold |
| `beta_warmup` | `int` | Epochs before KL annealing begins |
| `beta_anneal` | `int` | Epochs over which β ramps from `beta_start` → `beta_end` |
| `save_every` | `int` | Checkpoint interval (epochs) |
| `lr_warmup_epochs` | `int` | Epochs for linear LR ramp-up (0 → `lr`) |
| `es_patience` | `int` | Early stopping: max epochs without validation improvement |
| `es_min_epoch` | `int` | Early stopping: never stop before this epoch |
| `full_mnist` | `int` | 1 = load all 60K samples / 10 digits; 0 = digits 0–1 only |
| `data_dir` | `const char *` | Path to the directory containing the MNIST IDX files |
| `result_dir` | `const char *` | Directory for generated PGM output files |
| `model_dir` | `const char *` | Directory for checkpoint files |
| `model_file` | `const char *` | Full path to the checkpoint file |
| `version_tag` | `const char *` | Label printed in log output (`"v1"`, `"v2"`, `"v3"`) |

---

## Memory Model

Every `VAE` instance allocates a **single contiguous slab** on the heap at creation time. All weight matrices, activation buffers, gradient accumulators, and Adam moment buffers are carved out of this slab via pointer arithmetic.

```c
typedef struct VAE {
    VAEConfig cfg;   // runtime config — no compile-time #ifdefs
    Rng       rng;   // per-instance RNG — thread-safe by construction
    int       adam_t;

    float *_mem;     // ← single heap slab; every pointer below points into it

    float *enc_w1, *enc_b1;   // [enc_in × h1] / [h1]
    // ... activations, pre-activations, backward scratch,
    //     gradient accumulators, Adam m/v buffers
} VAE;
```

Benefits:
- **One allocation, one free** — no fragmentation, predictable teardown.
- **32-byte aligned** — slab is allocated via `aligned_alloc(32)` (one AVX2 register width). The compiler can emit `vmovaps` (aligned SIMD load) instead of `vmovups` in the GEMM inner loops, avoiding cache-line split penalties.
- **No stack overflow** — even the largest variant (~1.7M params) lives entirely on the heap.
- **Runtime integrity guard** — after all `NEXT()` pointer assignments, `create_vae` verifies `p == _mem + slab_size()` with an explicit `if / abort()`, not `assert()`. `assert()` is compiled out by `-DNDEBUG`; the runtime guard is always active and prints an actionable diagnostic with the byte-level delta.

`create_vae` returns `NULL` on allocation failure — it never calls `exit()`.

---

## Training Loop

The training loop in `vae_train.c` handles:

1. **Dataset split** — Fisher-Yates shuffle on load, then 90/10 train/val split.
2. **LR schedule** — linear warmup for `lr_warmup_epochs` epochs, then cosine decay to 5% of peak.
3. **KL annealing** — β held at `beta_start` during warmup, linearly ramped to `beta_end` over `beta_anneal` epochs.
4. **Batched validation** — full validation pass each epoch; val loss used for early stopping.
5. **Early stopping** — stops if val loss has not improved in `es_patience` epochs, but not before `es_min_epoch`.
6. **Checkpointing** — saves the model with the best validation loss; reloads on restart.

---

## RNG

The original implementation used a file-scope static for random state — a latent data race for any multi-threaded future. The RNG uses an owned struct:

```c
typedef struct { uint64_t state; int spare_ready; float spare; } Rng;

float rng_normal(Rng *r);    // Box-Muller — cached spare avoids wasted draws
float rng_uniform(Rng *r);   // xorshift-64
int   rng_int(Rng *r, int n);
```

Each `VAE` instance owns a `Rng`. Two models seeded differently produce fully independent noise — a prerequisite for correct data-parallel training.

---

## Optimizer

Adam is a **stateless helper**. The caller (`VAE`) owns the moment buffers and step counter. Gradient zeroing is the **sole responsibility of `vae_reset_grads()`**, which `memset`s all accumulator arrays before each batch. `adam_update()` reads the gradient and updates weights and moments — it does not zero the gradient, avoiding a hidden side effect.

```c
// Called once before each batch
vae_reset_grads(m);

// Then: forward → loss → backward → apply
vae_forward(m, xs, ls, bsz, /*training=*/1);
vae_backward(m, xs, ls, bsz, beta);
vae_apply_gradients(m, lr);   // calls adam_update for each layer
```

---

## Checkpoint Format

Checkpoints are stored as explicit **little-endian binary**, portable across architectures (x86, ARM, big-endian RISC):

```
[uint32 LE]  magic    = 0x45415643  ('CVAE')
[uint32 LE]  version  = 4
[float* LE]  weights: enc_w1 enc_b1 enc_w2 enc_b2 mu_w mu_b lv_w lv_b
                       dec_w1 dec_b1 dec_w2 dec_b2 dec_w3 dec_b3
[uint32 LE]  adam_t
[float* LE]  Adam m_ and v_ buffers (same layer order as weights)
```

Each float is written as 4 bytes in canonical LE order via `write_le_floats()` — host byte order is irrelevant. Old v3 checkpoints (raw `fwrite`) are rejected with an actionable message; re-train to produce a v4 checkpoint.

---

## API Call Sequence

The required per-batch call order is documented in `include/vae_model.h`:

```c
vae_reset_grads(m);                          // 1. zero accumulators
vae_forward(m, xs, labels, bsz, training=1); // 2. encoder + decoder
float loss = vae_loss(m, xs, bsz, beta);     // 3. ELBO (reads output)
vae_backward(m, xs, labels, bsz, beta);      // 4. accumulate grads
vae_apply_gradients(m, lr);                  // 5. Adam step
```

Key contracts (stated explicitly in each header):

| Invariant | Where documented |
|---|---|
| `vae_loss` / `vae_backward` must follow `vae_forward` for the same batch | `include/vae_model.h` |
| `vae_reset_grads` must precede each call to `vae_backward` | `include/vae_model.h` |
| `adam_update` does **not** zero `dw` — that is the caller's job | `include/vae_optimizer.h` |
| `load_model` requires a VAE created with a matching `VAEConfig` | `include/vae_io.h` |
| `train` requires `num_classes` to be consistent with dataset labels | `include/vae_train.h` |

For inference (no weight update), call only `vae_forward` + `vae_loss`; skip steps 1, 4, and 5.

---

## Test Suite

Tests use a zero-dependency header-only framework (`test_framework.h`). All tests link against the same implementation files as the production binary — no mocking.

```bash
make test
# Output:
# ALL 22765 TESTS PASSED   (exits 0; exits 1 on any failure)
```

| Suite | Tests |
|---|---|
| `test_rng` | Determinism across seeds; `[0,1)` range; normal mean/variance; `rng_int` bounds |
| `test_optimizer` | Adam convergence on scalar x²; `dw` is **not** zeroed by `adam_update` (documents the contract); vector convergence |
| `test_model` | Forward determinism (same seed); loss finite & positive; checkpoint save→load roundtrip; backward non-zero gradients; **numerical gradient check**; **bsz=1 edge case**; **extreme pixel values (0.0 / 1.0)** |
| `test_train` | Integration: loss strictly decreases over 3 epochs on synthetic data; **KL annealing schedule correctness** |

**Numerical gradient check** — the gold standard for backprop: compares `∂L/∂b₃[k]` from `vae_backward` against `(L(b₃[k]+ε) − L(b₃[k]−ε)) / 2ε` for four output-layer bias parameters, verifying the full chain rule end-to-end.

**bsz=1 edge case** — exercises per-sample pointer arithmetic and backward scratch buffers; off-by-one stride errors surface as wrong outputs or UBSan hits.

**Extreme pixel values** — verifies that the log-clamping guard in `vae_loss` (`p = vae_clamp(out, 1e-7, 1-1e-7)`) prevents `-Inf` loss when input pixels are exactly 0.0 or 1.0.

**KL annealing schedule** — replicated formula verified at five key points: warmup floor, midpoint, end-of-anneal, post-anneal clamp, and monotonicity over the full window.

---

## Build System

```makefile
CFLAGS  = -Wall -Wextra -O3 -std=c11 -march=native \
          -ffast-math -funroll-loops -ftree-vectorize \
          -flto -fomit-frame-pointer
HEADERS = $(wildcard include/*.h)
```

- `-march=native`: emits SIMD (SSE/AVX on x86, NEON on ARM) for the matrix loops via auto-vectorisation.
- `-flto`: cross-translation-unit inlining; the compiler can inline `linear_batch` into the training loop.
- `-ffast-math`: reassociation, FMA, no-NaN assumptions — consistent with the `isfinite` guards already in the loss.
- `asan` target: `-fsanitize=address,undefined -fno-omit-frame-pointer` for catching heap/stack overflows, use-after-free, signed overflow, and null dereferences during development.

**Tiled GEMM** — `linear_batch` iterates in `GEMM_TILE=64` blocks. A 64×64 `float` tile occupies 16 KB, fitting comfortably in a typical 32–64 KB L1 data cache. For each tile, `x[i]` values are reused across the full `j`-tile and `W[i][j…]` is a sequential row slice — reducing W cache-miss rate by O(bsz) versus the naive per-sample scan.

**`restrict` pointers** — all `linear_batch` and `linear_single` arguments are declared `restrict`, asserting to the compiler that the buffers do not alias. Combined with `-O3 -march=native -ftree-vectorize`, this is the signal GCC/Clang need to auto-vectorise the innermost dot-product loop.

**32-byte aligned slab** — `aligned_alloc(32)` guarantees that every `float*` carved from the slab is 32-byte aligned (one AVX2 register). With `restrict` already in place, the compiler can promote `vmovups` (potentially split across a cache line) to `vmovaps` (guaranteed single-line aligned load).

**Dependency tracking** — `$(HEADERS)` is listed as an explicit prerequisite for every target (main binaries and tests). Any change to a public header forces a full rebuild of every binary that transitively includes it, preventing stale-object bugs.

---

## Backward Pass (Chain Rule Derivations)

`vae_backward()` in `src/core/vae_model.c` loops over samples and accumulates gradients into the slab. Each section is annotated with the chain rule step it implements; here is the summary:

```
Forward graph (per sample):
  enc_in → [ELU] enc_h1 → [ELU] enc_h2 → mu, logvar
                                           ↓ reparameterisation
  z = mu + exp(0.5·lv) · ε,  ε ~ N(0,I)
  dec_in=[z|label] → [ELU] dec_h1 → [ELU] dec_h2 → [Sigmoid] output

Loss (ELBO):
  L = BCE(x, output) / IMAGE_SIZE  +  β · KL(q ∥ p) / latent
  KL = -0.5 · Σ_i [1 + lv_i - mu_i² - exp(lv_i)]
```

Backward sections (innermost to outermost):

| Section | Gradient computed | Formula |
|---|---|---|
| Output layer | `d_out[i]` | `(output_i − x_i) / (IMAGE_SIZE · bsz)` — BCE+sigmoid combined |
| dec_w3 / dec_b3 | weight/bias grads | outer product `dec_h2 ⊗ d_out` |
| dec_h2 (into ELU) | upstream signal | `W3ᵀ · d_out`, then `· ELU'(pre_dh2)` |
| dec_w2, dec_b2 | weight/bias grads | outer product `dec_h1 ⊗ d_pre_dh2` |
| dec_h1 (into ELU) | upstream signal | `W2ᵀ · d_pre_dh2`, then `· ELU'(pre_dh1)` |
| dec_w1, dec_b1 | weight/bias grads | outer product `dec_in ⊗ d_pre_dh1` |
| d_z | grad into latent | `W1ᵀ · d_pre_dh1` (label slots discarded) |
| Reparameterisation | `d_mu`, `d_lv` | `d_z + β·mu/(latent·bsz)` ; `d_z·0.5·σ·ε + β·0.5·(exp(lv)−1)/(latent·bsz)` |
| mu/lv heads | `d_muw`, `d_lvw` | outer products `enc_h2 ⊗ d_mu/d_lv` |
| enc_h2 (into ELU) | upstream signal | `mu_wᵀ·d_mu + lv_wᵀ·d_lv`, then `· ELU'(pre_eh2)` |
| enc_w2, enc_b2 | weight/bias grads | outer product `enc_h1 ⊗ d_pre_eh2` |
| enc_h1 (into ELU) | upstream signal | `enc_w2ᵀ · d_pre_eh2`, then `· ELU'(pre_eh1)` |
| enc_w1, enc_b1 | weight/bias grads | outer product `enc_in ⊗ d_pre_eh1` (input layer — no further upstream) |



## References

- Kingma & Welling, [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114) (2013)
- Higgins et al., [*β-VAE*](https://openreview.net/forum?id=Sy2fchgDl) — motivation for KL weighting and annealing
- LeCun et al., [MNIST database](http://yann.lecun.com/exdb/mnist/)

---

## Contributing

If you have recommendations or have spotted bugs, please don't hesitate to raise a pull request. See [CONTRIBUTING.md] for more details.
