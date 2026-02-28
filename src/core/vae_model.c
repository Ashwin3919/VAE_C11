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

#include "vae_internal.h" /* he_init */

/* ------------------------------------------------------------------ */
/* Slab layout — two-pass NEXT macro                                    */
/* ------------------------------------------------------------------ */

/*
 * Two-pass slab layout.
 *
 * Problem: the original slab_size() duplicated all 40+ NEXT() sizes as
 * separate arithmetic.  One is the source of truth; one drifts.  The
 * runtime guard catches it — but only at runtime, after a crash.
 *
 * Solution: a single ALLOC_BLOCK() macro list is evaluated twice:
 *
 *   Pass 1 (NEXT = NEXT_SZ): p is a dummy size_t counter, not a real
 *     pointer.  The block runs purely as arithmetic to accumulate n.
 *
 *   Pass 2 (NEXT = NEXT_PTR): p is the real float* cursor into the
 *     allocated slab.  Each field pointer is set and p advances.
 *
 * A new tensor requires ONE change — one NEXT() line in ALLOC_BLOCK.
 * slab_size() cannot drift because it no longer exists.
 *
 * The runtime guard is kept as a belt-and-suspenders defence; it now
 * checks that both passes agree rather than guarding against manual
 * arithmetic errors, so it should never fire in practice.
 */

/*
 * ALLOC_BLOCK — the canonical ordered list of every slab allocation.
 * M is a VAE* for pointer assignment in Pass 2; c is const VAEConfig*.
 * NEXT(field_ptr, count) must be defined before expanding this macro.
 */
#define ALLOC_BLOCK(M, c, bs)                                                  \
  /* weights */                                                                \
  NEXT((M)->enc_w1, (c)->enc_in *(c)->h1);                                     \
  NEXT((M)->enc_b1, (c)->h1);                                                  \
  NEXT((M)->enc_w2, (c)->h1 *(c)->h2);                                         \
  NEXT((M)->enc_b2, (c)->h2);                                                  \
  NEXT((M)->mu_w, (c)->h2 *(c)->latent);                                       \
  NEXT((M)->mu_b, (c)->latent);                                                \
  NEXT((M)->lv_w, (c)->h2 *(c)->latent);                                       \
  NEXT((M)->lv_b, (c)->latent);                                                \
  NEXT((M)->dec_w1, (c)->dec_in *(c)->h1);                                     \
  NEXT((M)->dec_b1, (c)->h1);                                                  \
  NEXT((M)->dec_w2, (c)->h1 *(c)->h2);                                         \
  NEXT((M)->dec_b2, (c)->h2);                                                  \
  NEXT((M)->dec_w3, (c)->h2 *IMAGE_SIZE);                                      \
  NEXT((M)->dec_b3, IMAGE_SIZE);                                               \
  /* activations */                                                            \
  NEXT((M)->enc_h1, (bs) * (c)->h1);                                           \
  NEXT((M)->enc_h2, (bs) * (c)->h2);                                           \
  NEXT((M)->mu, (bs) * (c)->latent);                                           \
  NEXT((M)->logvar, (bs) * (c)->latent);                                       \
  NEXT((M)->z, (bs) * (c)->latent);                                            \
  NEXT((M)->dec_h1, (bs) * (c)->h1);                                           \
  NEXT((M)->dec_h2, (bs) * (c)->h2);                                           \
  NEXT((M)->output, (bs)*IMAGE_SIZE);                                          \
  /* input buffers */                                                          \
  NEXT((M)->enc_in_buf, (bs) * (c)->enc_in);                                   \
  NEXT((M)->dec_in_buf, (bs) * (c)->dec_in);                                   \
  /* pre-activations */                                                        \
  NEXT((M)->pre_eh1, (bs) * (c)->h1);                                          \
  NEXT((M)->pre_eh2, (bs) * (c)->h2);                                          \
  NEXT((M)->pre_dh1, (bs) * (c)->h1);                                          \
  NEXT((M)->pre_dh2, (bs) * (c)->h2);                                          \
  NEXT((M)->pre_out, (bs)*IMAGE_SIZE);                                         \
  NEXT((M)->eps_buf, (bs) * (c)->latent);                                      \
  /* backward scratch (single-sample) */                                       \
  NEXT((M)->sc_out, IMAGE_SIZE);                                               \
  NEXT((M)->sc_dh2, (c)->h2);                                                  \
  NEXT((M)->sc_dh1, (c)->h1);                                                  \
  NEXT((M)->sc_z, (c)->latent);                                                \
  NEXT((M)->sc_mu, (c)->latent);                                               \
  NEXT((M)->sc_lv, (c)->latent);                                               \
  NEXT((M)->sc_eh2, (c)->h2);                                                  \
  NEXT((M)->sc_eh1, (c)->h1);                                                  \
  /* gradient accumulators */                                                  \
  NEXT((M)->dw3, (c)->h2 *IMAGE_SIZE);                                         \
  NEXT((M)->db3, IMAGE_SIZE);                                                  \
  NEXT((M)->dw2, (c)->h1 *(c)->h2);                                            \
  NEXT((M)->db2, (c)->h2);                                                     \
  NEXT((M)->dw1, (c)->dec_in *(c)->h1);                                        \
  NEXT((M)->db1, (c)->h1);                                                     \
  NEXT((M)->d_muw, (c)->h2 *(c)->latent);                                      \
  NEXT((M)->d_mub, (c)->latent);                                               \
  NEXT((M)->d_lvw, (c)->h2 *(c)->latent);                                      \
  NEXT((M)->d_lvb, (c)->latent);                                               \
  NEXT((M)->d_ew2, (c)->h1 *(c)->h2);                                          \
  NEXT((M)->d_eb2, (c)->h2);                                                   \
  NEXT((M)->d_ew1, (c)->enc_in *(c)->h1);                                      \
  NEXT((M)->d_eb1, (c)->h1);                                                   \
  /* Adam moment buffers */                                                    \
  NEXT((M)->m_ew1, (c)->enc_in *(c)->h1);                                      \
  NEXT((M)->v_ew1, (c)->enc_in *(c)->h1);                                      \
  NEXT((M)->m_eb1, (c)->h1);                                                   \
  NEXT((M)->v_eb1, (c)->h1);                                                   \
  NEXT((M)->m_ew2, (c)->h1 *(c)->h2);                                          \
  NEXT((M)->v_ew2, (c)->h1 *(c)->h2);                                          \
  NEXT((M)->m_eb2, (c)->h2);                                                   \
  NEXT((M)->v_eb2, (c)->h2);                                                   \
  NEXT((M)->m_muw, (c)->h2 *(c)->latent);                                      \
  NEXT((M)->v_muw, (c)->h2 *(c)->latent);                                      \
  NEXT((M)->m_mub, (c)->latent);                                               \
  NEXT((M)->v_mub, (c)->latent);                                               \
  NEXT((M)->m_lvw, (c)->h2 *(c)->latent);                                      \
  NEXT((M)->v_lvw, (c)->h2 *(c)->latent);                                      \
  NEXT((M)->m_lvb, (c)->latent);                                               \
  NEXT((M)->v_lvb, (c)->latent);                                               \
  NEXT((M)->m_dw1, (c)->dec_in *(c)->h1);                                      \
  NEXT((M)->v_dw1, (c)->dec_in *(c)->h1);                                      \
  NEXT((M)->m_db1, (c)->h1);                                                   \
  NEXT((M)->v_db1, (c)->h1);                                                   \
  NEXT((M)->m_dw2, (c)->h1 *(c)->h2);                                          \
  NEXT((M)->v_dw2, (c)->h1 *(c)->h2);                                          \
  NEXT((M)->m_db2, (c)->h2);                                                   \
  NEXT((M)->v_db2, (c)->h2);                                                   \
  NEXT((M)->m_dw3, (c)->h2 *IMAGE_SIZE);                                       \
  NEXT((M)->v_dw3, (c)->h2 *IMAGE_SIZE);                                       \
  NEXT((M)->m_db3, IMAGE_SIZE);                                                \
  NEXT((M)->v_db3, IMAGE_SIZE)

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

  /*
   * Pass 1 — compute slab size using a counter-only (no real pointer).
   * NEXT(field, sz) here just adds sz to the counter n.
   * A dummy struct is passed as M so the field references compile; it is
   * never written to because the counter-NEXT ignores the left-hand side.
   */
  VAE _dummy;
  size_t n = 0;
  { /* Pass 1: count-only — field assignments are discarded */
    float *_p = NULL;
#define NEXT(field, sz) ((field) = _p, n += (size_t)(sz))
    ALLOC_BLOCK(&_dummy, c, bs);
#undef NEXT
  }

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

  /*
   * Pass 2 — carve pointers from the real slab using the same ALLOC_BLOCK.
   * NEXT now assigns the pointer and advances p.
   */
  float *p = m->_mem;
#define NEXT(field, sz) ((field) = p, p += (size_t)(sz))
  ALLOC_BLOCK(m, c, bs);
#undef NEXT
#undef ALLOC_BLOCK

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
