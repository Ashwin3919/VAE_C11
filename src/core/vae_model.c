/*
 * vae_model.c — Conditional VAE: allocation, forward, backward, loss.
 *
 * All layer sizes are read from m->cfg at runtime.  No compile-time
 * H1/H2/LATENT macros.  Convention inside each function:
 *   const int h1 = m->cfg.h1, h2 = m->cfg.h2, ...;
 *
 * Memory layout: the _mem slab is allocated with aligned_alloc(SLAB_ALIGN)
 * so that every carved-out float* is 32-byte aligned.  This lets GCC/Clang
 * emit vmovaps (fast aligned load) instead of vmovups in the GEMM loops.
 *
 * Slab integrity is checked with an explicit runtime guard (not assert(),
 * which vanishes with -DNDEBUG in release builds).
 */

#include "vae_model.h"
#include "vae_math.h"
#include "vae_optimizer.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
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

/*
 * SLAB_ALIGN: byte alignment for the weight/activation slab.
 * 32 bytes = 256 bits = one AVX2 register (8 × float32).
 * Aligned loads let the compiler emit vmovaps (aligned) instead of
 * vmovups (unaligned), avoiding potential cache-line split penalties.
 * aligned_alloc() enforces this; the size is rounded up to a multiple
 * of SLAB_ALIGN so the C11 alignment contract is satisfied.
 */
#define SLAB_ALIGN 32

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

static void he_init(float *w, int fan_in, int fan_out, Rng *rng) {
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
static void linear_batch(float *restrict Y, const float *restrict X,
                         const float *restrict W, const float *restrict b,
                         int bsz, int in_n, int out_n) {
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
static void linear_single(float *restrict y, const float *restrict x,
                          const float *restrict W, const float *restrict b,
                          int in_n, int out_n) {
  for (int j = 0; j < out_n; j++)
    y[j] = b[j];
  for (int i = 0; i < in_n; i++) {
    const float xi = x[i];
    const float *restrict wr = W + (size_t)i * out_n;
    for (int j = 0; j < out_n; j++)
      y[j] += xi * wr[j];
  }
}

static void one_hot(float *out, int cls, int num_classes) {
  memset(out, 0, (size_t)num_classes * sizeof(float));
  if (cls >= 0 && cls < num_classes)
    out[cls] = 1.0f;
}

/* ------------------------------------------------------------------ */
/* Slab layout helper                                                   */
/* ------------------------------------------------------------------ */

/*
 * Compute the total number of floats required for the slab.
 * Must stay in sync with the NEXT() calls in create_vae().
 * A compile-time assert in create_vae() verifies this at runtime.
 */
static size_t slab_size(const VAEConfig *c) {
  const int bs = c->batch_size;
  size_t n = 0;
  /* weights */
  n += (size_t)c->enc_in * c->h1 + c->h1 + (size_t)c->h1 * c->h2 + c->h2 +
       2 * ((size_t)c->h2 * c->latent + c->latent) + (size_t)c->dec_in * c->h1 +
       c->h1 + (size_t)c->h1 * c->h2 + c->h2 + (size_t)c->h2 * IMAGE_SIZE +
       IMAGE_SIZE;
  /* activations (batch-sized) */
  n +=
      (size_t)bs * (c->h1 + c->h2 + 3 * c->latent + c->h1 + c->h2 + IMAGE_SIZE);
  /* input buffers */
  n += (size_t)bs * (c->enc_in + c->dec_in);
  /* pre-activation buffers (batch-sized) */
  n += (size_t)bs * (c->h1 + c->h2 + c->h1 + c->h2 + IMAGE_SIZE + c->latent);
  /* backward scratch (single-sample) */
  n += (size_t)(IMAGE_SIZE + c->h2 + c->h1 + 3 * c->latent + c->h2 + c->h1);
  /* gradient accumulators */
  n += (size_t)c->h2 * IMAGE_SIZE + IMAGE_SIZE + (size_t)c->h1 * c->h2 + c->h2 +
       (size_t)c->dec_in * c->h1 + c->h1 +
       2 * ((size_t)c->h2 * c->latent + c->latent) + (size_t)c->h1 * c->h2 +
       c->h2 + (size_t)c->enc_in * c->h1 + c->h1;
  /* Adam moment buffers (2× each weight/bias) */
  n += 2 * ((size_t)c->enc_in * c->h1 + c->h1 + (size_t)c->h1 * c->h2 + c->h2 +
            2 * ((size_t)c->h2 * c->latent + c->latent) +
            (size_t)c->dec_in * c->h1 + c->h1 + (size_t)c->h1 * c->h2 + c->h2 +
            (size_t)c->h2 * IMAGE_SIZE + IMAGE_SIZE);
  return n;
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                            */
/* ------------------------------------------------------------------ */

VAE *create_vae(const VAEConfig *cfg, uint64_t rng_seed) {
  VAE *m = calloc(1, sizeof(VAE));
  if (!m) {
    fprintf(stderr, "[ERROR] calloc VAE struct failed\n");
    return NULL;
  }

  m->cfg = *cfg;
  m->adam_t = 0;
  rng_init(&m->rng, rng_seed);

  const VAEConfig *c = &m->cfg;
  const int bs = c->batch_size;
  size_t n = slab_size(c);

  /* Round up byte count to a multiple of SLAB_ALIGN — required by
   * aligned_alloc. */
  size_t byte_sz = n * sizeof(float);
  size_t aligned_sz = (byte_sz + SLAB_ALIGN - 1) & ~(size_t)(SLAB_ALIGN - 1);

  /*
   * aligned_alloc(32) guarantees every float* carved from the slab is
   * 32-byte aligned (one AVX2 register).  The compiler can then emit
   * vmovaps (aligned load) in the GEMM inner loop instead of the slower
   * vmovups (unaligned), avoiding possible cache-line split penalties.
   */
  m->_mem = aligned_alloc(SLAB_ALIGN, aligned_sz);
  if (!m->_mem) {
    fprintf(stderr, "[ERROR] aligned_alloc VAE slab failed (size=%zu)\n",
            aligned_sz);
    free(m);
    return NULL;
  }
  memset(m->_mem, 0, aligned_sz);

  float *p = m->_mem;
#define NEXT(ptr, sz) ((ptr) = p, p += (size_t)(sz))

  /* weights */
  NEXT(m->enc_w1, c->enc_in * c->h1);
  NEXT(m->enc_b1, c->h1);
  NEXT(m->enc_w2, c->h1 * c->h2);
  NEXT(m->enc_b2, c->h2);
  NEXT(m->mu_w, c->h2 * c->latent);
  NEXT(m->mu_b, c->latent);
  NEXT(m->lv_w, c->h2 * c->latent);
  NEXT(m->lv_b, c->latent);
  NEXT(m->dec_w1, c->dec_in * c->h1);
  NEXT(m->dec_b1, c->h1);
  NEXT(m->dec_w2, c->h1 * c->h2);
  NEXT(m->dec_b2, c->h2);
  NEXT(m->dec_w3, c->h2 * IMAGE_SIZE);
  NEXT(m->dec_b3, IMAGE_SIZE);
  /* activations */
  NEXT(m->enc_h1, bs * c->h1);
  NEXT(m->enc_h2, bs * c->h2);
  NEXT(m->mu, bs * c->latent);
  NEXT(m->logvar, bs * c->latent);
  NEXT(m->z, bs * c->latent);
  NEXT(m->dec_h1, bs * c->h1);
  NEXT(m->dec_h2, bs * c->h2);
  NEXT(m->output, bs * IMAGE_SIZE);
  /* input buffers */
  NEXT(m->enc_in_buf, bs * c->enc_in);
  NEXT(m->dec_in_buf, bs * c->dec_in);
  /* pre-activations */
  NEXT(m->pre_eh1, bs * c->h1);
  NEXT(m->pre_eh2, bs * c->h2);
  NEXT(m->pre_dh1, bs * c->h1);
  NEXT(m->pre_dh2, bs * c->h2);
  NEXT(m->pre_out, bs * IMAGE_SIZE);
  NEXT(m->eps_buf, bs * c->latent);
  /* backward scratch (single-sample) */
  NEXT(m->sc_out, IMAGE_SIZE);
  NEXT(m->sc_dh2, c->h2);
  NEXT(m->sc_dh1, c->h1);
  NEXT(m->sc_z, c->latent);
  NEXT(m->sc_mu, c->latent);
  NEXT(m->sc_lv, c->latent);
  NEXT(m->sc_eh2, c->h2);
  NEXT(m->sc_eh1, c->h1);
  /* gradient accumulators */
  NEXT(m->dw3, c->h2 * IMAGE_SIZE);
  NEXT(m->db3, IMAGE_SIZE);
  NEXT(m->dw2, c->h1 * c->h2);
  NEXT(m->db2, c->h2);
  NEXT(m->dw1, c->dec_in * c->h1);
  NEXT(m->db1, c->h1);
  NEXT(m->d_muw, c->h2 * c->latent);
  NEXT(m->d_mub, c->latent);
  NEXT(m->d_lvw, c->h2 * c->latent);
  NEXT(m->d_lvb, c->latent);
  NEXT(m->d_ew2, c->h1 * c->h2);
  NEXT(m->d_eb2, c->h2);
  NEXT(m->d_ew1, c->enc_in * c->h1);
  NEXT(m->d_eb1, c->h1);
  /* Adam moment buffers */
  NEXT(m->m_ew1, c->enc_in * c->h1);
  NEXT(m->v_ew1, c->enc_in * c->h1);
  NEXT(m->m_eb1, c->h1);
  NEXT(m->v_eb1, c->h1);
  NEXT(m->m_ew2, c->h1 * c->h2);
  NEXT(m->v_ew2, c->h1 * c->h2);
  NEXT(m->m_eb2, c->h2);
  NEXT(m->v_eb2, c->h2);
  NEXT(m->m_muw, c->h2 * c->latent);
  NEXT(m->v_muw, c->h2 * c->latent);
  NEXT(m->m_mub, c->latent);
  NEXT(m->v_mub, c->latent);
  NEXT(m->m_lvw, c->h2 * c->latent);
  NEXT(m->v_lvw, c->h2 * c->latent);
  NEXT(m->m_lvb, c->latent);
  NEXT(m->v_lvb, c->latent);
  NEXT(m->m_dw1, c->dec_in * c->h1);
  NEXT(m->v_dw1, c->dec_in * c->h1);
  NEXT(m->m_db1, c->h1);
  NEXT(m->v_db1, c->h1);
  NEXT(m->m_dw2, c->h1 * c->h2);
  NEXT(m->v_dw2, c->h1 * c->h2);
  NEXT(m->m_db2, c->h2);
  NEXT(m->v_db2, c->h2);
  NEXT(m->m_dw3, c->h2 * IMAGE_SIZE);
  NEXT(m->v_dw3, c->h2 * IMAGE_SIZE);
  NEXT(m->m_db3, IMAGE_SIZE);
  NEXT(m->v_db3, IMAGE_SIZE);
#undef NEXT

  /*
   * Slab integrity check — runtime guard, not assert().
   *
   * assert() is compiled out by -DNDEBUG in release builds, so the check
   * would silently disappear exactly when it matters most.  Instead we use
   * an explicit if-abort that is always present, costs one pointer compare
   * per model creation, and prints an actionable diagnostic.
   */
  if (p != m->_mem + n) {
    fprintf(stderr,
            "[FATAL] Slab layout mismatch in create_vae(): "
            "expected end=%p  actual p=%p  (delta=%+td floats).\n"
            "  → update slab_size() to match the NEXT() calls above.\n",
            (void *)(m->_mem + n), (void *)p, (ptrdiff_t)(p - (m->_mem + n)));
    free(m->_mem);
    free(m);
    abort();
  }

  /* Weight initialisation (He) */
  he_init(m->enc_w1, c->enc_in, c->h1, &m->rng);
  he_init(m->enc_w2, c->h1, c->h2, &m->rng);
  he_init(m->mu_w, c->h2, c->latent, &m->rng);
  he_init(m->lv_w, c->h2, c->latent, &m->rng);
  he_init(m->dec_w1, c->dec_in, c->h1, &m->rng);
  he_init(m->dec_w2, c->h1, c->h2, &m->rng);
  he_init(m->dec_w3, c->h2, IMAGE_SIZE, &m->rng);
  for (int i = 0; i < c->h2 * c->latent; i++)
    m->lv_w[i] *= 0.01f;

  long pc = (long)c->enc_in * c->h1 + c->h1 + (long)c->h1 * c->h2 + c->h2 +
            2 * ((long)c->h2 * c->latent + c->latent) +
            (long)c->dec_in * c->h1 + c->h1 + (long)c->h1 * c->h2 + c->h2 +
            (long)c->h2 * IMAGE_SIZE + IMAGE_SIZE;
  printf("[INFO] CVAE %s: enc_in=%d  latent=%d  classes=%d  params=%ldK\n",
         c->version_tag, c->enc_in, c->latent, c->num_classes, pc / 1000);
  return m;
}

void free_vae(VAE *m) {
  if (!m)
    return;
  free(m->_mem);
  free(m);
}

/* ------------------------------------------------------------------ */
/* Forward pass                                                         */
/* ------------------------------------------------------------------ */

void vae_forward(VAE *m, float **xs, const int *labels, int bsz, int training) {
  const VAEConfig *c = &m->cfg;
  const int h1 = c->h1, h2 = c->h2, latent = c->latent;
  const int enc_in = c->enc_in, dec_in = c->dec_in, nc = c->num_classes;

  /* Build enc_in_buf[bsz, enc_in] */
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

  /* Reparametrisation */
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

  /* Build dec_in_buf[bsz, dec_in] */
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

  /* Output */
  linear_batch(m->pre_out, m->dec_h2, m->dec_w3, m->dec_b3, bsz, h2,
               IMAGE_SIZE);
  for (int i = 0; i < bsz * IMAGE_SIZE; i++)
    m->output[i] = vae_sigmoid(m->pre_out[i]);
}

/* ------------------------------------------------------------------ */
/* Loss                                                                 */
/* ------------------------------------------------------------------ */

float vae_loss(VAE *m, float **xs, int bsz, float beta) {
  const int latent = m->cfg.latent;
  float total = 0.0f;
  for (int s = 0; s < bsz; s++) {
    const float *x = xs[s];
    const float *out = m->output + (size_t)s * IMAGE_SIZE;
    const float *mu = m->mu + (size_t)s * latent;
    const float *lv = m->logvar + (size_t)s * latent;

    float recon = 0.0f;
    for (int i = 0; i < IMAGE_SIZE; i++) {
      float p = vae_clamp(out[i], 1e-7f, 1.0f - 1e-7f);
      recon -= x[i] * logf(p) + (1.0f - x[i]) * logf(1.0f - p);
    }
    recon /= IMAGE_SIZE;

    float kl = 0.0f;
    for (int i = 0; i < latent; i++)
      kl += 1.0f + lv[i] - mu[i] * mu[i] - expf(lv[i]);
    kl = -0.5f * kl / (float)latent;

    float t = recon + beta * kl;
    total += isfinite(t) ? t : 1.0f;
  }
  return total / (float)bsz;
}

/* ------------------------------------------------------------------ */
/* Gradient helpers                                                     */
/* ------------------------------------------------------------------ */

void vae_reset_grads(VAE *m) {
  const VAEConfig *c = &m->cfg;
#define CLR(ptr, sz) memset((ptr), 0, (size_t)(sz) * sizeof(float))
  CLR(m->dw3, c->h2 * IMAGE_SIZE);
  CLR(m->db3, IMAGE_SIZE);
  CLR(m->dw2, c->h1 * c->h2);
  CLR(m->db2, c->h2);
  CLR(m->dw1, c->dec_in * c->h1);
  CLR(m->db1, c->h1);
  CLR(m->d_muw, c->h2 * c->latent);
  CLR(m->d_mub, c->latent);
  CLR(m->d_lvw, c->h2 * c->latent);
  CLR(m->d_lvb, c->latent);
  CLR(m->d_ew2, c->h1 * c->h2);
  CLR(m->d_eb2, c->h2);
  CLR(m->d_ew1, c->enc_in * c->h1);
  CLR(m->d_eb1, c->h1);
#undef CLR
}

/* ------------------------------------------------------------------ */
/* Backward pass                                                        */
/* ------------------------------------------------------------------ */

void vae_backward(VAE *m, float **xs, const int *labels, int bsz, float beta) {
  (void)labels; /* labels already embedded in enc_in_buf/dec_in_buf */
  const VAEConfig *c = &m->cfg;
  const int h1 = c->h1, h2 = c->h2, latent = c->latent;
  const int enc_in = c->enc_in, dec_in = c->dec_in;
  const float scale = 1.0f / (float)bsz;

  /* Slab scratch aliases */
  float *d_out = m->sc_out;
  float *d_dh2 = m->sc_dh2;
  float *d_dh1 = m->sc_dh1;
  float *d_z = m->sc_z;
  float *d_mu = m->sc_mu;
  float *d_lv = m->sc_lv;
  float *d_eh2 = m->sc_eh2;
  float *d_eh1 = m->sc_eh1;

  for (int s = 0; s < bsz; s++) {
    const float *x = xs[s];
    const float *out_s = m->output + (size_t)s * IMAGE_SIZE;
    const float *dec_h2s = m->dec_h2 + (size_t)s * h2;
    const float *dec_h1s = m->dec_h1 + (size_t)s * h1;
    const float *enc_h2s = m->enc_h2 + (size_t)s * h2;
    const float *enc_h1s = m->enc_h1 + (size_t)s * h1;
    const float *mu_s = m->mu + (size_t)s * latent;
    const float *lv_s = m->logvar + (size_t)s * latent;
    const float *eps_s = m->eps_buf + (size_t)s * latent;
    const float *pdh2_s = m->pre_dh2 + (size_t)s * h2;
    const float *pdh1_s = m->pre_dh1 + (size_t)s * h1;
    const float *peh2_s = m->pre_eh2 + (size_t)s * h2;
    const float *peh1_s = m->pre_eh1 + (size_t)s * h1;
    const float *enc_in_s = m->enc_in_buf + (size_t)s * enc_in;
    const float *dec_in_s = m->dec_in_buf + (size_t)s * dec_in;

    /* ── Decoder output layer ──────────────────────────────────────────
     * Loss:  L_recon = -Σ_i [ x_i·log(p_i) + (1-x_i)·log(1-p_i) ] / IMAGE_SIZE
     * Output activation: p_i = sigmoid(pre_out_i)
     *
     * Combined BCE + sigmoid gradient (cancels sigmoid denominator):
     *   dL/d_pre_out_i = p_i - x_i      (per sample)
     *
     * Scaled by 1/IMAGE_SIZE (from loss normalisation) and 1/bsz (batch mean):
     *   d_out[i] = (out_i - x_i) / (IMAGE_SIZE · bsz)
     *
     * Weight gradient (outer product, accumulated over batch):
     *   dL/dw3[i,j] += dec_h2[i] · d_out[j]
     *
     * Bias gradient (same signal, no upstream activation):
     *   dL/db3[j]   += d_out[j]
     *
     * Upstream gradient into dec_h2 (chain rule through W3^T):
     *   d_dh2[i] = Σ_j  dec_w3[i,j] · d_out[j]
     */
    for (int i = 0; i < IMAGE_SIZE; i++) {
      d_out[i] = (out_s[i] - x[i]) / ((float)IMAGE_SIZE * (float)bsz);
      m->db3[i] += d_out[i];
    }
    memset(d_dh2, 0, (size_t)h2 * sizeof(float));
    for (int i = 0; i < h2; i++)
      for (int j = 0; j < IMAGE_SIZE; j++) {
        m->dw3[(size_t)i * IMAGE_SIZE + j] += dec_h2s[i] * d_out[j];
        d_dh2[i] += m->dec_w3[(size_t)i * IMAGE_SIZE + j] * d_out[j];
      }

    /* ── Decoder layer 2 (dec_h2 = ELU(pre_dh2)) ──────────────────────
     * Chain rule through ELU:
     *   dL/d_pre_dh2[i] = dL/d_dec_h2[i] · ELU'(pre_dh2[i])
     *   where ELU'(x) = 1 if x > 0, else ELU_ALPHA·exp(x)
     *
     * Weight gradient:   dL/dw2[i,j] += dec_h1[i] · d_pre_dh2[j]
     * Bias gradient:     dL/db2[j]   += d_pre_dh2[j]
     * Upstream (W2^T):   d_dh1[i]     = Σ_j dec_w2[i,j] · d_pre_dh2[j]
     */
    for (int i = 0; i < h2; i++)
      d_dh2[i] *= vae_elu_d(pdh2_s[i]);
    memset(d_dh1, 0, (size_t)h1 * sizeof(float));
    for (int j = 0; j < h2; j++)
      m->db2[j] += d_dh2[j];
    for (int i = 0; i < h1; i++)
      for (int j = 0; j < h2; j++) {
        m->dw2[(size_t)i * h2 + j] += dec_h1s[i] * d_dh2[j];
        d_dh1[i] += m->dec_w2[(size_t)i * h2 + j] * d_dh2[j];
      }

    /* ── Decoder layer 1 (dec_h1 = ELU(pre_dh1)) ──────────────────────
     * Same pattern as layer 2:
     *   dL/d_pre_dh1[i] = dL/d_dec_h1[i] · ELU'(pre_dh1[i])
     *
     * dec_in_buf = [z | one_hot(label)]; label slots have zero upstream grad.
     * Weight gradient:   dL/dw1[i,j] += dec_in[i] · d_pre_dh1[j]
     * Bias gradient:     dL/db1[j]   += d_pre_dh1[j]
     * Upstream into z:   d_z[i]       = Σ_j dec_w1[i,j] · d_pre_dh1[j]
     *                                   (only i < latent; label slots
     * discarded)
     */
    for (int i = 0; i < h1; i++)
      d_dh1[i] *= vae_elu_d(pdh1_s[i]);
    for (int j = 0; j < h1; j++)
      m->db1[j] += d_dh1[j];
    memset(d_z, 0, (size_t)latent * sizeof(float));
    for (int i = 0; i < dec_in; i++)
      for (int j = 0; j < h1; j++) {
        m->dw1[(size_t)i * h1 + j] += dec_in_s[i] * d_dh1[j];
        if (i < latent)
          d_z[i] += m->dec_w1[(size_t)i * h1 + j] * d_dh1[j];
      }

    /* ── Reparameterisation trick ───────────────────────────────────────
     * Forward:  z_i = mu_i + exp(0.5·lv_i) · eps_i    (eps ~ N(0,I))
     * KL loss:  L_KL = -0.5·Σ_i [1 + lv_i - mu_i² - exp(lv_i)] / latent
     *
     * d_z already carries the reconstruction gradient (scaled by 1/bsz
     * from the output layer above).  KL gradients get an additional
     * `scale = 1/bsz` factor to match the batch-mean convention.
     *
     * dL/d_mu_i = dL/d_z_i · ∂z_i/∂mu_i  +  dL_KL/d_mu_i
     *           = d_z[i]                   +  beta·mu_i / latent · scale
     *
     * dL/d_lv_i = dL/d_z_i · ∂z_i/∂lv_i  +  dL_KL/d_lv_i
     *           = d_z[i] · 0.5·σ_i·eps_i  +  beta·0.5·(exp(lv_i)-1)/latent ·
     * scale where σ_i = exp(0.5·lv_i)
     */
    for (int i = 0; i < latent; i++) {
      float sig = expf(0.5f * lv_s[i]);
      d_mu[i] = d_z[i] + (beta * mu_s[i] / (float)latent) * scale;
      d_lv[i] = d_z[i] * 0.5f * sig * eps_s[i] +
                (beta * 0.5f * (expf(lv_s[i]) - 1.0f) / (float)latent) * scale;
      m->d_mub[i] += d_mu[i];
      m->d_lvb[i] += d_lv[i];
    }

    /* ── Encoder μ / log σ² heads (linear, no activation) ─────────────
     * dL/d_mu_w[i,j]  += enc_h2[i] · d_mu[j]
     * dL/d_lv_w[i,j]  += enc_h2[i] · d_lv[j]
     *
     * Upstream into enc_h2 receives contributions from both heads:
     *   d_eh2[i] = Σ_j ( mu_w[i,j]·d_mu[j] + lv_w[i,j]·d_lv[j] )
     */
    memset(d_eh2, 0, (size_t)h2 * sizeof(float));
    for (int i = 0; i < h2; i++)
      for (int j = 0; j < latent; j++) {
        m->d_muw[(size_t)i * latent + j] += enc_h2s[i] * d_mu[j];
        m->d_lvw[(size_t)i * latent + j] += enc_h2s[i] * d_lv[j];
        d_eh2[i] += m->mu_w[(size_t)i * latent + j] * d_mu[j] +
                    m->lv_w[(size_t)i * latent + j] * d_lv[j];
      }

    /* ── Encoder layer 2 (enc_h2 = ELU(pre_eh2)) ──────────────────────
     * dL/d_pre_eh2[i] = d_eh2[i] · ELU'(pre_eh2[i])
     *
     * Weight gradient:   dL/d_ew2[i,j] += enc_h1[i] · d_pre_eh2[j]
     * Bias gradient:     dL/d_eb2[j]   += d_pre_eh2[j]
     * Upstream (ew2^T):  d_eh1[i]       = Σ_j enc_w2[i,j] · d_pre_eh2[j]
     */
    for (int i = 0; i < h2; i++)
      d_eh2[i] *= vae_elu_d(peh2_s[i]);
    memset(d_eh1, 0, (size_t)h1 * sizeof(float));
    for (int j = 0; j < h2; j++)
      m->d_eb2[j] += d_eh2[j];
    for (int i = 0; i < h1; i++)
      for (int j = 0; j < h2; j++) {
        m->d_ew2[(size_t)i * h2 + j] += enc_h1s[i] * d_eh2[j];
        d_eh1[i] += m->enc_w2[(size_t)i * h2 + j] * d_eh2[j];
      }

    /* ── Encoder layer 1 (enc_h1 = ELU(pre_eh1)) ──────────────────────
     * dL/d_pre_eh1[i] = d_eh1[i] · ELU'(pre_eh1[i])
     *
     * Weight gradient:   dL/d_ew1[i,j] += enc_in[i] · d_pre_eh1[j]
     * Bias gradient:     dL/d_eb1[j]   += d_pre_eh1[j]
     * (enc_in = [image | one_hot]; no upstream past the input layer)
     */
    for (int i = 0; i < h1; i++)
      d_eh1[i] *= vae_elu_d(peh1_s[i]);
    for (int j = 0; j < h1; j++)
      m->d_eb1[j] += d_eh1[j];
    for (int i = 0; i < enc_in; i++)
      for (int j = 0; j < h1; j++)
        m->d_ew1[(size_t)i * h1 + j] += enc_in_s[i] * d_eh1[j];
  }
}

/* ------------------------------------------------------------------ */
/* Gradient application                                                 */
/* ------------------------------------------------------------------ */

void vae_apply_gradients(VAE *m, float lr) {
  m->adam_t++;
  int t = m->adam_t;
  const VAEConfig *c = &m->cfg;
#define AU(w, dw, mm, v, n) adam_update((w), (dw), (mm), (v), (n), lr, t)
  AU(m->dec_w3, m->dw3, m->m_dw3, m->v_dw3, c->h2 * IMAGE_SIZE);
  AU(m->dec_b3, m->db3, m->m_db3, m->v_db3, IMAGE_SIZE);
  AU(m->dec_w2, m->dw2, m->m_dw2, m->v_dw2, c->h1 * c->h2);
  AU(m->dec_b2, m->db2, m->m_db2, m->v_db2, c->h2);
  AU(m->dec_w1, m->dw1, m->m_dw1, m->v_dw1, c->dec_in * c->h1);
  AU(m->dec_b1, m->db1, m->m_db1, m->v_db1, c->h1);
  AU(m->mu_w, m->d_muw, m->m_muw, m->v_muw, c->h2 * c->latent);
  AU(m->mu_b, m->d_mub, m->m_mub, m->v_mub, c->latent);
  AU(m->lv_w, m->d_lvw, m->m_lvw, m->v_lvw, c->h2 * c->latent);
  AU(m->lv_b, m->d_lvb, m->m_lvb, m->v_lvb, c->latent);
  AU(m->enc_w2, m->d_ew2, m->m_ew2, m->v_ew2, c->h1 * c->h2);
  AU(m->enc_b2, m->d_eb2, m->m_eb2, m->v_eb2, c->h2);
  AU(m->enc_w1, m->d_ew1, m->m_ew1, m->v_ew1, c->enc_in * c->h1);
  AU(m->enc_b1, m->d_eb1, m->m_eb1, m->v_eb1, c->h1);
#undef AU
}

/* ------------------------------------------------------------------ */
/* Decoder-only pass (for generation — bypasses encoder)               */
/* vae_generate.c calls this externally via the header's internal API  */
/* ------------------------------------------------------------------ */

void vae_decode(VAE *m, const float *z, int label) {
  const VAEConfig *c = &m->cfg;
  const int h1 = c->h1, h2 = c->h2, dec_in = c->dec_in;

  memcpy(m->dec_in_buf, z, (size_t)c->latent * sizeof(float));
  one_hot(m->dec_in_buf + c->latent, label, c->num_classes);

  linear_single(m->pre_dh1, m->dec_in_buf, m->dec_w1, m->dec_b1, dec_in, h1);
  for (int i = 0; i < h1; i++)
    m->dec_h1[i] = vae_elu(m->pre_dh1[i]);

  linear_single(m->pre_dh2, m->dec_h1, m->dec_w2, m->dec_b2, h1, h2);
  for (int i = 0; i < h2; i++)
    m->dec_h2[i] = vae_elu(m->pre_dh2[i]);

  linear_single(m->pre_out, m->dec_h2, m->dec_w3, m->dec_b3, h2, IMAGE_SIZE);
  for (int i = 0; i < IMAGE_SIZE; i++)
    m->output[i] = vae_sigmoid(m->pre_out[i]);
}
