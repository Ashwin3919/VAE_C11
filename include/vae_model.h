/*
 * vae_model.h — Conditional VAE model: allocation, forward, backward, loss.
 *
 * All layer sizes come from VAEConfig at runtime; no compile-time #ifdefs.
 * The VAE struct owns a single contiguous memory slab for all float arrays.
 * Each VAE instance carries its own Rng, making concurrent use thread-safe.
 */
#ifndef VAE_MODEL_H
#define VAE_MODEL_H

#include "vae_config.h"
#include "vae_rng.h"

/* ------------------------------------------------------------------ */
/* VAE struct                                                           */
/* ------------------------------------------------------------------ */
typedef struct VAE {
  VAEConfig cfg; /* copy of runtime config (h1, h2, latent, …) */
  Rng rng;       /* per-model RNG — thread-safe                  */
  int adam_t;    /* shared Adam step counter                     */

  float *_mem; /* single slab — all pointers below point here  */

  /* weights */
  float *enc_w1, *enc_b1; /* [enc_in, h1]    / [h1]       */
  float *enc_w2, *enc_b2; /* [h1,     h2]    / [h2]       */
  float *mu_w, *mu_b;     /* [h2,  latent]   / [latent]   */
  float *lv_w, *lv_b;     /* [h2,  latent]   / [latent]   */
  float *dec_w1, *dec_b1; /* [dec_in, h1]    / [h1]       */
  float *dec_w2, *dec_b2; /* [h1,     h2]    / [h2]       */
  float *dec_w3, *dec_b3; /* [h2, IMAGE_SIZE]/ [IMAGE_SIZE]*/

  /* activations — each row dimension is batch_size */
  float *enc_h1, *enc_h2;
  float *mu, *logvar, *z;
  float *dec_h1, *dec_h2, *output;

  /* input concat buffers [batch_size × enc_in/dec_in] */
  float *enc_in_buf, *dec_in_buf;

  /* pre-activation buffers [batch_size × layer_width] */
  float *pre_eh1, *pre_eh2;
  float *pre_dh1, *pre_dh2, *pre_out;
  float *eps_buf;

  /* backward scratch — single-sample, reused per sample in the bwd loop */
  float *sc_out;          /* IMAGE_SIZE                             */
  float *sc_dh2, *sc_dh1; /* h2, h1                                 */
  float *sc_z;            /* latent                                 */
  float *sc_mu, *sc_lv;   /* latent                                 */
  float *sc_eh2, *sc_eh1; /* h2, h1                                 */

  /* gradient accumulators (reset each batch, weight-shaped) */
  float *dw3, *db3;
  float *dw2, *db2;
  float *dw1, *db1;
  float *d_muw, *d_mub;
  float *d_lvw, *d_lvb;
  float *d_ew2, *d_eb2;
  float *d_ew1, *d_eb1;

  /* Adam moment buffers (2× each weight/bias) */
  float *m_ew1, *v_ew1, *m_eb1, *v_eb1;
  float *m_ew2, *v_ew2, *m_eb2, *v_eb2;
  float *m_muw, *v_muw, *m_mub, *v_mub;
  float *m_lvw, *v_lvw, *m_lvb, *v_lvb;
  float *m_dw1, *v_dw1, *m_db1, *v_db1;
  float *m_dw2, *v_dw2, *m_db2, *v_db2;
  float *m_dw3, *v_dw3, *m_db3, *v_db3;
} VAE;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                            */
/* ------------------------------------------------------------------ */

/* Allocate and initialise a VAE from config; seed the per-model RNG. */
VAE *create_vae(const VAEConfig *cfg, uint64_t rng_seed);

/* Release the model (single free for the slab + one for the struct). */
void free_vae(VAE *m);

/* ------------------------------------------------------------------ */
/* Forward / loss / backward                                           */
/* ------------------------------------------------------------------ */
/*
 * Required call sequence per training batch:
 *
 *   vae_reset_grads(m);                          // 1. zero accumulators
 *   vae_forward(m, xs, ls, bsz, training=1);     // 2. encoder + decoder
 *   float loss = vae_loss(m, xs, bsz, beta);     // 3. ELBO (reads output)
 *   vae_backward(m, xs, ls, bsz, beta);          // 4. accumulate grads
 *   vae_apply_gradients(m, lr);                  // 5. Adam step
 *
 * Invariants:
 *   - vae_loss()     MUST be called after vae_forward() for the same batch.
 *   - vae_backward() MUST be called after vae_forward() for the same batch
 *                    (it reads activations stored in the slab by vae_forward).
 *   - vae_reset_grads() MUST be called before each new batch; gradients are
 *                    accumulated (+=) by vae_backward, never overwritten.
 *   - vae_apply_gradients() MUST NOT be called if vae_backward() was skipped
 *                    (e.g. when vae_loss returns a non-finite value).
 *   - For inference (no gradient update), call only vae_forward() then
 *                    vae_loss(); skip steps 1, 4, and 5 entirely.
 */

/*
 * vae_forward — batched encoder + decoder.
 *   xs[bsz]     : pointers to IMAGE_SIZE pixel buffers.
 *   labels[bsz] : digit class for each sample (0 … num_classes-1).
 *   training    : 1 = sample z via reparameterisation; 0 = use mean (μ).
 *
 * Writes activations into the slab (enc_h1, enc_h2, mu, logvar, z,
 * dec_h1, dec_h2, output) and saves eps_buf for use by vae_backward.
 * Precondition: bsz <= cfg.batch_size.
 */
void vae_forward(VAE *m, float **xs, const int *labels, int bsz, int training);

/*
 * vae_loss — average ELBO loss over the batch.
 *   Precondition: vae_forward() has been called for this batch.
 *   Returns mean over bsz of: BCE(x, x̂)/IMAGE_SIZE + beta·KL/latent.
 *   Non-finite per-sample terms are replaced with 1.0 (NaN guard).
 */
float vae_loss(VAE *m, float **xs, int bsz, float beta);

/*
 * vae_backward — accumulate gradients into m->dw*, m->db*, m->d_*.
 *   Precondition: vae_forward() has been called for this batch.
 *   Scale factor (1/bsz) is applied internally; do not pre-scale xs.
 *   Gradients are accumulated (+=); call vae_reset_grads() first.
 *   Call vae_apply_gradients() once after to commit the Adam step.
 */
void vae_backward(VAE *m, float **xs, const int *labels, int bsz, float beta);

/* ------------------------------------------------------------------ */
/* Gradient application                                                 */
/* ------------------------------------------------------------------ */

/*
 * vae_reset_grads — zero all gradient accumulators in the slab.
 *   MUST be called once before each batch, before vae_backward().
 *   Not calling this causes gradients to accumulate across batches.
 */
void vae_reset_grads(VAE *m);

/*
 * vae_apply_gradients — run Adam on every parameter tensor, then
 *   increment adam_t.  Reads the accumulated gradients written by
 *   vae_backward(); does NOT zero them (call vae_reset_grads() next batch).
 */
void vae_apply_gradients(VAE *m, float lr);

/* ------------------------------------------------------------------ */
/* Generation                                                           */
/* ------------------------------------------------------------------ */

/*
 * vae_decode — decoder-only forward pass, bypasses encoder.
 *   z[latent]  : latent vector to decode.
 *   label      : conditioning class (0 … num_classes-1).
 * Writes m->output[IMAGE_SIZE] in place (single sample, bsz=1 path).
 */
void vae_decode(VAE *m, const float *z, int label);

#endif /* VAE_MODEL_H */
