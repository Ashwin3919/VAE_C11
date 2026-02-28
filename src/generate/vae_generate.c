/*
 * vae_generate.c — decoder-only generation.
 */
#include "vae_generate.h"
#include "vae_math.h"

#include <math.h>
#include <stdio.h>

/* vae_decode() is implemented in vae_model.c and declared here locally. */
void vae_decode(VAE *m, const float *z, int label);

void generate_digit(VAE *m, int label, float temp, int epoch, int sample_idx) {
  const VAEConfig *c = &m->cfg;
  float z[c->latent];
  for (int i = 0; i < c->latent; i++)
    z[i] = rng_normal(&m->rng) * temp;

  vae_decode(m, z, label);

  char path[PATH_BUF_SIZE];
  snprintf(path, sizeof path, "%s/epoch_%03d_digit%d_s%d.pgm", c->result_dir,
           epoch, label, sample_idx);
  FILE *f = fopen(path, "w");
  if (!f)
    return;
  fprintf(f, "P2\n28 28\n255\n");
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++)
      fprintf(f, "%d ", (int)(m->output[y * 28 + x] * 255));
    fprintf(f, "\n");
  }
  fclose(f);
}

void generate_epoch_samples(VAE *m, int epoch) {
  const VAEConfig *c = &m->cfg;
  for (int cls = 0; cls < c->num_classes; cls++)
    for (int s = 0; s < 3; s++)
      generate_digit(m, cls, 0.8f, epoch, s);
  printf("[EPOCH %3d] samples -> %s/epoch_%03d_digit*_s*.pgm\n", epoch,
         c->result_dir, epoch);
}
