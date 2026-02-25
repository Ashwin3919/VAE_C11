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
  c.beta_start = 0.00001f;
  c.beta_end = 0.0005f;
  c.grad_clip = GRAD_CLIP_DEFAULT;
  c.beta_warmup = 100;
  c.beta_anneal = 150;
  c.save_every = 50;
  c.lr_warmup_epochs = 30;
  c.es_patience = 60;
  c.es_min_epoch = 150;
  c.full_mnist = 0;
  c.version_tag = "v1";
  c.data_dir = "data";
  c.result_dir = "results_main/v1";
  c.model_dir = "models";
  c.model_file = "models/vae_v1.bin";
  return c;
}

VAEConfig vae_config_v2(void) {
  VAEConfig c = {0};
  c.h1 = 512;
  c.h2 = 256;
  c.latent = 64;
  c.num_classes = 2;
  c.enc_in = IMAGE_SIZE + c.num_classes;
  c.dec_in = c.latent + c.num_classes;
  c.batch_size = 64;
  c.epochs = 400;
  c.lr = 0.001f;
  c.beta_start = 0.00001f;
  c.beta_end = 0.0005f;
  c.grad_clip = GRAD_CLIP_DEFAULT;
  c.beta_warmup = 100;
  c.beta_anneal = 150;
  c.save_every = 50;
  c.lr_warmup_epochs = 30;
  c.es_patience = 60;
  c.es_min_epoch = 150;
  c.full_mnist = 0;
  c.version_tag = "v2";
  c.data_dir = "data";
  c.result_dir = "results_main/v2";
  c.model_dir = "models";
  c.model_file = "models/vae_v2.bin";
  return c;
}

VAEConfig vae_config_v3(void) {
  VAEConfig c = {0};
  c.h1 = 640;
  c.h2 = 320;
  c.latent = 128;
  c.num_classes = 10;
  c.enc_in = IMAGE_SIZE + c.num_classes;
  c.dec_in = c.latent + c.num_classes;
  c.batch_size = 64;
  c.epochs = 800;
  c.lr = 0.0008f;
  c.beta_start = 0.00001f;
  c.beta_end = 0.0005f;
  c.grad_clip = GRAD_CLIP_DEFAULT;
  c.beta_warmup = 100;
  c.beta_anneal = 150;
  c.save_every = 50;
  c.lr_warmup_epochs = 30;
  c.es_patience = 60;
  c.es_min_epoch = 150;
  c.full_mnist = 1;
  c.version_tag = "v3";
  c.data_dir = "data";
  c.result_dir = "results_main/v3";
  c.model_dir = "models";
  c.model_file = "models/vae_v3.bin";
  return c;
}
