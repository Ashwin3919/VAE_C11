/*
 * vae_math.h — numeric constants and inline activation functions.
 *
 * All magic numbers that appeared scattered through the original code are
 * collected here as named constants so they are easy to find and change.
 */
#ifndef VAE_MATH_H
#define VAE_MATH_H

#include <math.h>
#include <stdint.h>

/* ── image / data ────────────────────────────────────────── */
#define IMAGE_SIZE 784 /* 28 × 28 MNIST pixels   */
/* PATH_BUF_SIZE lives in vae_config.h (filesystem paths are not math). */

/* ── activation bounds ───────────────────────────────────── */
#define ELU_ALPHA 0.2f
#define LOGVAR_MIN (-10.0f)
#define LOGVAR_MAX 4.0f
#define SIGMOID_CLAMP 15.0f
#define ELU_CLAMP (-15.0f) /* lower bound for expf()  */

/* ── gradient clipping ───────────────────────────────────── */
#define GRAD_CLIP_DEFAULT 5.0f

/* ── math ────────────────────────────────────────────────── */
#ifndef M_PI_F
#define M_PI_F 3.14159265f
#endif

/* ── inline helpers ──────────────────────────────────────── */
static inline float vae_clamp(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

static inline float vae_elu(float x) {
  return x > 0.0f ? x
                  : ELU_ALPHA * (expf(vae_clamp(x, ELU_CLAMP, 0.0f)) - 1.0f);
}

static inline float vae_elu_d(float pre) {
  return pre > 0.0f ? 1.0f : ELU_ALPHA * expf(vae_clamp(pre, ELU_CLAMP, 0.0f));
}

static inline float vae_sigmoid(float x) {
  x = vae_clamp(x, -SIGMOID_CLAMP, SIGMOID_CLAMP);
  return 1.0f / (1.0f + expf(-x));
}

static inline float vae_clip_grad(float g, float clip) {
  if (!isfinite(g))
    return 0.0f;
  return vae_clamp(g, -clip, clip);
}

#endif /* VAE_MATH_H */
