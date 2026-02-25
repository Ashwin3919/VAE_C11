/*
 * vae_rng.c — thread-safe xorshift64 + Box-Muller RNG.
 */
#include "vae_rng.h"
#include <math.h>

void rng_init(Rng *r, uint64_t seed) {
  r->state = seed ^ 0x123456789ABCULL;
  if (!r->state)
    r->state = 1; /* 0 is a degenerate xorshift state */
  r->spare_ready = 0;
  r->spare = 0.0f;
}

float rng_uniform(Rng *r) {
  r->state ^= r->state << 13;
  r->state ^= r->state >> 7;
  r->state ^= r->state << 17;
  return (float)(r->state & 0xFFFFFF) / (float)0x1000000; /* [0, 1) */
}

float rng_normal(Rng *r) {
  if (r->spare_ready) {
    r->spare_ready = 0;
    return r->spare;
  }
  float u, v, s;
  do {
    u = rng_uniform(r) * 2.0f - 1.0f;
    v = rng_uniform(r) * 2.0f - 1.0f;
    s = u * u + v * v;
  } while (s >= 1.0f || s == 0.0f);
  s = sqrtf(-2.0f * logf(s) / s);
  r->spare = v * s;
  r->spare_ready = 1;
  return u * s;
}

int rng_int(Rng *r, int n) { return (int)(rng_uniform(r) * (float)n) % n; }
