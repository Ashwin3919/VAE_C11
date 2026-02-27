/*
 * vae_loss.c — ELBO loss, Adam parameter update, and decoder-only generation.
 *
 * Separation rationale:
 *   vae_model.c    – slab allocation / lifecycle (create_vae, free_vae)
 *   vae_forward.c  – encoder + decoder forward inference
 *   vae_backward.c – gradient accumulation (vae_backward, vae_reset_grads)
 *   vae_loss.c     – ELBO loss, Adam update, generation decode   ← this file
 */

#include "vae_math.h"
#include "vae_model.h"
#include "vae_optimizer.h"

#include <math.h>
#include <string.h>

#include "vae_internal.h" /* linear_single, one_hot — defined in vae_forward.c */

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* Loss                                                                 */
/* ------------------------------------------------------------------ */

/*
 * vae_loss — average ELBO over the batch.
 *
 * ELBO = E[log p(x|z)] − β · KL(q(z|x) || p(z))
 *
 * We minimise −ELBO, so:
 *   L = BCE(x, x̂)/IMAGE_SIZE  +  β·KL/latent
 *
 * BCE per pixel: −[x·log(p) + (1−x)·log(1−p)], clamped to avoid log(0).
 * KL per dimension: 0.5·(exp(lv) + mu² − 1 − lv).
 *
 * Non-finite per-sample terms are replaced by 1.0 (NaN guard) so that a
 * single saturated sample cannot corrupt the entire batch.
 */
float vae_loss(VAE *m, float **xs, int bsz, float beta) {
  const int latent = m->cfg.latent;
  float total = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total) schedule(static)
#endif
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
/* Gradient application (Adam)                                          */
/* ------------------------------------------------------------------ */

/*
 * vae_apply_gradients — run one Adam step on every parameter tensor.
 *
 * Adam implementation:
 *   m_t = β₁·m_{t-1} + (1−β₁)·g            (first moment, clipped gradient)
 *   v_t = β₂·v_{t-1} + (1−β₂)·g²           (second moment)
 *   θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)  (bias-corrected update)
 *
 * Gradient clipping is applied before the moment updates (in adam_update),
 * so extreme individual gradients cannot destabilise the moment estimates.
 *
 * AMSGrad / weight-decay are intentionally omitted: the training loss is
 * well-conditioned on MNIST with the existing ELBO formulation, and the
 * added complexity has shown no measurable benefit in ablations.  Adding
 * AMSGrad would replace v[i] with max(v_max[i], v[i]) requiring an extra
 * slab of size equal to all parameters — a ~2× memory cost for zero gain.
 */
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
/* Decoder-only pass (generation — bypasses encoder)                   */
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
