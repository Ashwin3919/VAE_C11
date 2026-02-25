/*
 * main.c — entry point (~30 lines).
 *
 * Select the model preset (v1/v2/v3) at compile time via -DVERSION_V2 /
 * -DVERSION_V3; the architecture parameters live entirely in VAEConfig, not in
 * macros.
 */
#include "vae_config.h"
#include "vae_generate.h"
#include "vae_io.h"
#include "vae_math.h"
#include "vae_model.h"
#include "vae_train.h"

#include <stdio.h>
#include <time.h>

int main(void) {
#if defined(VERSION_V3)
  VAEConfig cfg = vae_config_v3();
#elif defined(VERSION_V2)
  VAEConfig cfg = vae_config_v2();
#else
  VAEConfig cfg = vae_config_v1();
#endif

  printf(
      "CVAE %s  enc=[%d]->%d->%d->[%dz]  dec=[%dz]->%d->%d->[%d]  classes=%d\n",
      cfg.version_tag, cfg.enc_in, cfg.h1, cfg.h2, cfg.latent, cfg.dec_in,
      cfg.h1, cfg.h2, IMAGE_SIZE, cfg.num_classes);

  VAE *m = create_vae(&cfg, (uint64_t)time(NULL));
  if (!m) {
    fprintf(stderr, "[FATAL] out of memory — cannot create model\n");
    return 1;
  }
  load_model(m, cfg.model_file); /* resume if checkpoint exists */

  Dataset *ds = load_dataset(&cfg);
  if (!ds) {
    free_vae(m);
    return 1;
  }

  train(m, ds);

  printf("\n[GEN] final samples per digit...\n");
  for (int cls = 0; cls < cfg.num_classes; cls++)
    for (int s = 0; s < 3; s++)
      generate_digit(m, cls, 0.8f, 9999, s);
  printf("[GEN] done -> %s/epoch_9999_digit*_s*.pgm\n", cfg.result_dir);

  free_dataset(ds);
  free_vae(m);
  return 0;
}
