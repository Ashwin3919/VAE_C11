/*
 * vae_config.h — runtime model configuration (replaces #ifdef V1/V2/V3).
 *
 * Create a VAEConfig with one of the preset constructors, then pass a pointer
 * to create_vae().  You can also populate the struct directly for custom sizes.
 */
#ifndef VAE_CONFIG_H
#define VAE_CONFIG_H

/*
 * PATH_BUF_SIZE: maximum filesystem path length for checkpoint/result files.
 * Placed here (not vae_math.h) because it is a configuration/IO constant,
 * not a numeric or activation-function constant.
 */
#define PATH_BUF_SIZE 512

typedef struct {
  /* Architecture */
  int h1, h2, latent, num_classes;

  /* Derived (set by vae_config_*) — do not set manually */
  int enc_in; /* IMAGE_SIZE + num_classes */
  int dec_in; /* latent    + num_classes */

  /* Training hyper-parameters */
  int batch_size, epochs;
  float lr, beta_start, beta_end, grad_clip;
  int beta_warmup, beta_anneal, save_every;
  int lr_warmup_epochs; /* linear LR ramp: 0→lr over this many epochs       */
  int es_patience;      /* early stopping: max epochs without val improvement */
  int es_min_epoch;     /* early stopping: never stop before this epoch       */
  int full_mnist;       /* 1 = load all 10 digits, 0 = digits 0-1 only */

  /* Paths */
  const char *version_tag;
  const char *data_dir; /* root directory for MNIST binary files */
  const char *result_dir;
  const char *model_file;
  const char *model_dir;
} VAEConfig;

/* Preset constructors */
VAEConfig vae_config_v1(void); /* 784→256→128→z32   binary digits  */
VAEConfig vae_config_v2(void); /* 784→512→256→z64   binary digits  */
VAEConfig vae_config_v3(void); /* 784→640→320→z128  all 10 digits  */

#endif /* VAE_CONFIG_H */
