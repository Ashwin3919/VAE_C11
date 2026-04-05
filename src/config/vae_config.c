/*
 * vae_config.c — VAEConfig preset constructors.
 */
#include "vae_config.h"
#include "vae_math.h" /* IMAGE_SIZE */

VAEConfig vae_config_v1(void) {
  VAEConfig c = {0};
  c.h1 = 256;
  c.h2 = 128;
  c.latent = 32;
  c.num_classes = 2;
  c.enc_in = IMAGE_SIZE + c.num_classes;
  c.dec_in = c.latent + c.num_classes;
  c.batch_size = 64;
  c.epochs = 300;
  c.lr = 0.001f;
  c.beta_start = 0.001f;  /* start higher — reduces initial jump from x2000 to x50 */
  c.beta_end = 0.05f;     /* 0.2 over-regularised; 0.05 balances recon vs KL for MNIST */
  c.grad_clip = GRAD_CLIP_DEFAULT;
  c.beta_warmup = 50;
  c.beta_anneal = 100;  /* ramp finishes at epoch 150 */
  c.save_every = 50;
  c.lr_warmup_epochs = 30;
  c.es_patience = 60;
  c.es_min_epoch = 220; /* was 150 — must be > beta_warmup+beta_anneal+buffer so
                           early stop doesn't trigger on pre-KL best_val */
  c.full_mnist = 0;
  c.version_tag = "v1";
  c.data_dir = "data";
  c.result_dir = "results_main/v1";
  c.model_dir = "models";
  c.model_file = "models/vae_v1.bin";
  return c;
}


VAEConfig vae_config_v3(void) {
  VAEConfig c = {0};
  /* Same hidden dims as v1 — v1 proved 256/128 is sufficient for MNIST.
   * Only latent is doubled (32->64) to give room for 5x more classes.
   * Keeps speed ~1500 img/s and avoids the instability of the old 640/320 arch. */
  c.h1 = 256;
  c.h2 = 128;
  c.latent = 64;
  c.num_classes = 10;
  c.enc_in = IMAGE_SIZE + c.num_classes;
  c.dec_in = c.latent + c.num_classes;
  c.batch_size = 64;
  c.epochs = 400;
  c.lr = 0.0001f;       /* v3 has 5x more batches/epoch than v1 (843 vs 178).
                           Model converges by epoch 2 (~1700 steps). LR above
                           0.0001 disrupts a nearly-converged model. */
  c.beta_start = 0.001f;  /* same logic as v1 */
  c.beta_end = 0.05f;
  c.grad_clip = GRAD_CLIP_DEFAULT;
  c.beta_warmup = 50;
  c.beta_anneal = 100;  /* ramp finishes at epoch 150 */
  c.save_every = 50;
  c.lr_warmup_epochs = 5;
  c.es_patience = 60;
  c.es_min_epoch = 220; /* same logic as v1 */
  c.full_mnist = 1;
  c.version_tag = "v3";
  c.data_dir = "data";
  c.result_dir = "results_main/v3";
  c.model_dir = "models";
  c.model_file = "models/vae_v3.bin";
  return c;
}
