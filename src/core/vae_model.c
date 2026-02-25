/*
 * vae_model.c — Conditional VAE: slab lifecycle only.
 *
 * This file has one responsibility: allocating and freeing the VAE struct
 * and its memory slab.  All computation is in dedicated files:
 *
 *   vae_forward.c  — encoder + decoder forward pass
 *   vae_backward.c — gradient accumulation (vae_backward, vae_reset_grads)
 *   vae_loss.c     — ELBO loss, Adam update, generation decode (vae_decode)
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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * SLAB_ALIGN: byte alignment for the weight/activation slab.
 * 32 bytes = 256 bits = one AVX2 register (8 × float32).
 * aligned_alloc() enforces this; size is rounded up to a multiple of
 * SLAB_ALIGN so the C11 alignment contract is satisfied.
 */
#define SLAB_ALIGN 32

/* Forward declarations for helpers defined in vae_forward.c */
void he_init(float *w, int fan_in, int fan_out, Rng *rng);

/* ------------------------------------------------------------------ */
/* Slab layout helper                                                   */
/* ------------------------------------------------------------------ */

/*
 * slab_size — total number of floats required for the VAE slab.
 * Must stay in sync with the NEXT() calls in create_vae().
 * A runtime guard in create_vae() catches any mismatch.
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
    m->lv_w[i] *=
        0.01f; /* small logvar head init — prevents posterior collapse */

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
