/*
 * vae_optimizer.c — Adam optimiser.
 */
#include "vae_optimizer.h"
#include "vae_math.h"
#include <math.h>

void adam_update(float *w, float *dw, float *m, float *v, int n, float lr,
                 int t) {
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  const float bc1 = 1.0f - powf(b1, (float)t);
  const float bc2 = 1.0f - powf(b2, (float)t);
  for (int i = 0; i < n; i++) {
    float g = vae_clip_grad(dw[i], GRAD_CLIP_DEFAULT);
    m[i] = b1 * m[i] + (1.0f - b1) * g;
    v[i] = b2 * v[i] + (1.0f - b2) * g * g;
    w[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps);
  }
}
