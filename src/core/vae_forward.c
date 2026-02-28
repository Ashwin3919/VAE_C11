/*
 * vae_forward.c — VAE forward pass.
 *
 * Implements vae_forward() (encoder + reparameterisation + decoder) together
 * with the internal GEMM helpers and weight initialisation used only here.
 *
 * Separation rationale:
 *   vae_model.c    – slab allocation / lifecycle (create_vae, free_vae)
 *   vae_forward.c  – encoder + decoder forward inference         ← this file
 *   vae_backward.c – gradient accumulation (vae_backward, vae_reset_grads)
 *   vae_loss.c     – ELBO loss, Adam update, generation decode
 */

#include "vae_internal.h" /* canonical signatures for he_init, linear_batch, linear_single, one_hot */
#include "vae_math.h"
#include "vae_model.h"

#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * GEMM_TILE: blocking factor for the tiled matrix multiply.
 * A GEMM_TILE × GEMM_TILE sub-matrix of W occupies
 *   64 × 64 × 4 = 16 384 bytes (16 KB),
 * which fits comfortably in a typical 32–64 KB L1 data cache.
 * Keeping that tile hot while accumulating all bsz sample rows
 * reduces W cache-miss rate by O(bsz) versus the naive per-sample scan.
 */
#define GEMM_TILE 64

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

void he_init(float *w, int fan_in, int fan_out, Rng *rng) {
  float s = sqrtf(2.0f / (float)fan_in);
  for (int i = 0; i < fan_in * fan_out; i++)
    w[i] = rng_normal(rng) * s;
}

/*
 * linear_batch — tiled batched GEMM: Y[bsz×out] = X[bsz×in] * W[in×out] +
 * b[out]
 *
 * Tiling: iterate (i-tile, j-tile) blocks so a GEMM_TILE×GEMM_TILE sub-matrix
 * of W stays in L1 cache while all bsz sample rows are accumulated against it.
 *
 * restrict: asserts to the compiler that Y, X, W, b are non-aliased pointers.
 * Combined with -O3 -march=native -ftree-vectorize this lets GCC/Clang emit
 * SIMD instructions (SSE/AVX) for the innermost dot-product loop.
 *
 * OpenMP: each sample row writes to a distinct slice of Y and reads distinct
 * rows of X, so the bsz loop parallelises with zero data races.  Compiled
 * out when _OPENMP is not defined, so the binary builds without -fopenmp.
 */
void linear_batch(float *restrict Y, const float *restrict X,
                  const float *restrict W, const float *restrict b, int bsz,
                  int in_n, int out_n) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int s = 0; s < bsz; s++) {
    float *restrict y = Y + (size_t)s * out_n;
    const float *restrict x = X + (size_t)s * in_n;

    /* Bias initialisation for this output row */
    for (int j = 0; j < out_n; j++)
      y[j] = b[j];

    /* Tiled GEMM: 64×64 W-tile = 16 KB fits in L1 (≥32 KB on modern CPUs).
     * For each tile, x[i] values are re-used across the full j-tile,
     * and wr[j] is a contiguous row-slice of W — sequential read. */
    for (int i0 = 0; i0 < in_n; i0 += GEMM_TILE) {
      const int i_end = (i0 + GEMM_TILE < in_n) ? i0 + GEMM_TILE : in_n;
      for (int j0 = 0; j0 < out_n; j0 += GEMM_TILE) {
        const int j_end = (j0 + GEMM_TILE < out_n) ? j0 + GEMM_TILE : out_n;
        const int jlen = j_end - j0;
        float *restrict ys = y + j0;
        for (int i = i0; i < i_end; i++) {
          const float xi = x[i];
          const float *restrict wr = W + (size_t)i * out_n + j0;
          for (int j = 0; j < jlen; j++)
            ys[j] += xi * wr[j];
        }
      }
    }
  }
}

/* Single-sample linear (generation path — no batching, no OpenMP overhead). */
void linear_single(float *restrict y, const float *restrict x,
                   const float *restrict W, const float *restrict b, int in_n,
                   int out_n) {
  for (int j = 0; j < out_n; j++)
    y[j] = b[j];
  for (int i = 0; i < in_n; i++) {
    const float xi = x[i];
    const float *restrict wr = W + (size_t)i * out_n;
    for (int j = 0; j < out_n; j++)
      y[j] += xi * wr[j];
  }
}

void one_hot(float *out, int cls, int num_classes) {
  memset(out, 0, (size_t)num_classes * sizeof(float));
  if (cls >= 0 && cls < num_classes)
    out[cls] = 1.0f;
}

/* ------------------------------------------------------------------ */
/* Forward pass                                                         */
/* ------------------------------------------------------------------ */

void vae_forward(VAE *m, float **xs, const int *labels, int bsz, int training) {
  const VAEConfig *c = &m->cfg;
  const int h1 = c->h1, h2 = c->h2, latent = c->latent;
  const int enc_in = c->enc_in, dec_in = c->dec_in, nc = c->num_classes;

  /* Build enc_in_buf[bsz, enc_in] = [image | one_hot(label)] */
  for (int s = 0; s < bsz; s++) {
    float *dst = m->enc_in_buf + (size_t)s * enc_in;
    memcpy(dst, xs[s], IMAGE_SIZE * sizeof(float));
    one_hot(dst + IMAGE_SIZE, labels[s], nc);
  }

  /* Encoder layer 1 */
  linear_batch(m->pre_eh1, m->enc_in_buf, m->enc_w1, m->enc_b1, bsz, enc_in,
               h1);
  for (int i = 0; i < bsz * h1; i++)
    m->enc_h1[i] = vae_elu(m->pre_eh1[i]);

  /* Encoder layer 2 */
  linear_batch(m->pre_eh2, m->enc_h1, m->enc_w2, m->enc_b2, bsz, h1, h2);
  for (int i = 0; i < bsz * h2; i++)
    m->enc_h2[i] = vae_elu(m->pre_eh2[i]);

  /* mu and logvar heads */
  linear_batch(m->mu, m->enc_h2, m->mu_w, m->mu_b, bsz, h2, latent);
  linear_batch(m->logvar, m->enc_h2, m->lv_w, m->lv_b, bsz, h2, latent);
  for (int i = 0; i < bsz * latent; i++)
    m->logvar[i] = vae_clamp(m->logvar[i], LOGVAR_MIN, LOGVAR_MAX);

  /* Reparameterisation: z = mu + exp(0.5*logvar) * eps */
  for (int s = 0; s < bsz; s++) {
    float *mu_s = m->mu + (size_t)s * latent;
    float *lv_s = m->logvar + (size_t)s * latent;
    float *z_s = m->z + (size_t)s * latent;
    float *eps_s = m->eps_buf + (size_t)s * latent;
    for (int i = 0; i < latent; i++) {
      float eps = training ? rng_normal(&m->rng) : 0.0f;
      eps_s[i] = eps;
      z_s[i] = mu_s[i] + expf(0.5f * lv_s[i]) * eps;
    }
  }

  /* Build dec_in_buf[bsz, dec_in] = [z | one_hot(label)] */
  for (int s = 0; s < bsz; s++) {
    float *dst = m->dec_in_buf + (size_t)s * dec_in;
    memcpy(dst, m->z + (size_t)s * latent, (size_t)latent * sizeof(float));
    one_hot(dst + latent, labels[s], nc);
  }

  /* Decoder layer 1 */
  linear_batch(m->pre_dh1, m->dec_in_buf, m->dec_w1, m->dec_b1, bsz, dec_in,
               h1);
  for (int i = 0; i < bsz * h1; i++)
    m->dec_h1[i] = vae_elu(m->pre_dh1[i]);

  /* Decoder layer 2 */
  linear_batch(m->pre_dh2, m->dec_h1, m->dec_w2, m->dec_b2, bsz, h1, h2);
  for (int i = 0; i < bsz * h2; i++)
    m->dec_h2[i] = vae_elu(m->pre_dh2[i]);

  /* Output layer: sigmoid(W3·dec_h2 + b3) */
  linear_batch(m->pre_out, m->dec_h2, m->dec_w3, m->dec_b3, bsz, h2,
               IMAGE_SIZE);
  for (int i = 0; i < bsz * IMAGE_SIZE; i++)
    m->output[i] = vae_sigmoid(m->pre_out[i]);
}
