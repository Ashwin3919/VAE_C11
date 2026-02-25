/*
 * vae_rng.h — thread-safe pseudo-random number generator.
 *
 * Each Rng instance is independent; passing separate instances to concurrent
 * training runs eliminates the data race on the original file-scope statics.
 */
#ifndef VAE_RNG_H
#define VAE_RNG_H

#include <stdint.h>

typedef struct {
  uint64_t state;
  int spare_ready;
  float spare;
} Rng;

/* Seed the generator (state=0 is replaced with 1 to avoid degenerate cycle). */
void rng_init(Rng *r, uint64_t seed);

/* Uniform sample in [0, 1). */
float rng_uniform(Rng *r);

/* Standard-normal sample via Box-Muller (state kept inside *r). */
float rng_normal(Rng *r);

/* Uniform integer in [0, n-1]. */
int rng_int(Rng *r, int n);

#endif /* VAE_RNG_H */
