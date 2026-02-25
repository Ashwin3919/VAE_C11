/*
 * vae_internal.h — private cross-TU declarations for src/core/*.c
 *
 * These symbols are defined in vae_forward.c but used by vae_loss.c
 * (and potentially vae_backward.c in the future).  Putting them here
 * instead of repeating forward declarations in each .c file means:
 *
 *   1. A signature change in vae_forward.c only needs to be updated here;
 *      the compiler will catch any mismatch in all consuming TUs.
 *   2. There is one canonical source of truth for internal ABI.
 *
 * This header is intentionally NOT installed in include/ — it describes
 * a private implementation contract between the core source files only.
 * External consumers see only the public API in include/vae_model.h.
 */
#ifndef VAE_INTERNAL_H
#define VAE_INTERNAL_H

#include "vae_rng.h" /* Rng */

/*
 * he_init — initialise a weight matrix with He (Kaiming) normal scaling.
 *   w        : flat array of (fan_in × fan_out) floats
 *   fan_in   : number of input connections
 *   fan_out  : number of output neurons (used only to size the loop)
 *   rng      : per-model RNG (thread-safe; caller owns it)
 */
void he_init(float *w, int fan_in, int fan_out, Rng *rng);

/*
 * linear_batch — tiled batched GEMM: Y[bsz×out] = X[bsz×in] * W[in×out] +
 * b[out] Parallelised with OpenMP when _OPENMP is defined. All pointer
 * arguments carry `restrict` — callers must not alias them.
 */
void linear_batch(float *restrict Y, const float *restrict X,
                  const float *restrict W, const float *restrict b, int bsz,
                  int in_n, int out_n);

/*
 * linear_single — single-sample linear layer (generation path, no OMP
 * overhead).
 */
void linear_single(float *restrict y, const float *restrict x,
                   const float *restrict W, const float *restrict b, int in_n,
                   int out_n);

/*
 * one_hot — write a one-hot vector of length num_classes into out[].
 *   out[cls] = 1.0; all other entries = 0.0.
 *   If cls is out of range [0, num_classes), all entries are set to 0.
 */
void one_hot(float *out, int cls, int num_classes);

#endif /* VAE_INTERNAL_H */
