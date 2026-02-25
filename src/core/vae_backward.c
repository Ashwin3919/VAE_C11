/*
 * vae_backward.c — VAE backward pass and gradient reset.
 *
 * Implements vae_reset_grads() and vae_backward().
 * Reads activations written by vae_forward() from the shared slab.
 *
 * Separation rationale:
 *   vae_model.c    – slab allocation / lifecycle (create_vae, free_vae)
 *   vae_forward.c  – encoder + decoder forward inference
 *   vae_backward.c – gradient accumulation                     ← this file
 *   vae_loss.c     – ELBO loss, Adam update, generation decode
 *
 * Gradient derivations are commented inline.  The key invariants:
 *   – Gradients are ACCUMULATED (+=); call vae_reset_grads() before each batch.
 *   – All scale-by-(1/bsz) terms are applied here, not in the caller.
 *   – The reparameterisation trick uses eps_buf saved by vae_forward().
 */

#include "vae_math.h"
#include "vae_model.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* Forward declarations for helpers defined in vae_forward.c           */
/* ------------------------------------------------------------------ */
/* (linear_batch / linear_single are not needed in the backward pass;
 *  gradient accumulation uses hand-written outer-product loops to keep
 *  the dependency on vae_forward.c minimal.) */

/* ------------------------------------------------------------------ */
/* Gradient reset                                                       */
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

  /* Slab scratch aliases (single-sample, reused per loop iteration) */
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
     * Combined BCE + sigmoid gradient (sigmoid denominator cancels):
     *   dL/d_pre_out_i = p_i - x_i      (per sample)
     *
     * Scaled by 1/IMAGE_SIZE (loss normalisation) and 1/bsz (batch mean):
     *   d_out[i] = (out_i - x_i) / (IMAGE_SIZE · bsz)
     *
     * Weight gradient:  dL/dw3[i,j] += dec_h2[i] · d_out[j]
     * Bias gradient:    dL/db3[j]   += d_out[j]
     * Upstream (W3^T):  d_dh2[i]     = Σ_j dec_w3[i,j] · d_out[j]
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
     * Weight gradient:  dL/dw2[i,j] += dec_h1[i] · d_pre_dh2[j]
     * Bias gradient:    dL/db2[j]   += d_pre_dh2[j]
     * Upstream (W2^T):  d_dh1[i]     = Σ_j dec_w2[i,j] · d_pre_dh2[j]
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
     * Weight gradient:  dL/dw1[i,j] += dec_in[i] · d_pre_dh1[j]
     * Bias gradient:    dL/db1[j]   += d_pre_dh1[j]
     * Upstream into z:  d_z[i]       = Σ_j dec_w1[i,j] · d_pre_dh1[j]
     *                                  (only i < latent; label slots discarded)
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
     * dL/d_mu_i = d_z[i]  +  beta·mu_i / latent · scale
     *
     * dL/d_lv_i = d_z[i] · 0.5·σ_i·eps_i
     *           + beta·0.5·(exp(lv_i)−1) / latent · scale
     *   where σ_i = exp(0.5·lv_i)
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
     * Upstream into enc_h2 from both heads:
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
     * Weight gradient:  dL/d_ew2[i,j] += enc_h1[i] · d_pre_eh2[j]
     * Bias gradient:    dL/d_eb2[j]   += d_pre_eh2[j]
     * Upstream (ew2^T): d_eh1[i]       = Σ_j enc_w2[i,j] · d_pre_eh2[j]
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
     * Weight gradient:  dL/d_ew1[i,j] += enc_in[i] · d_pre_eh1[j]
     * Bias gradient:    dL/d_eb1[j]   += d_pre_eh1[j]
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
