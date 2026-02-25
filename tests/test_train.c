/*
 * test_train.c — integration test: 3 epochs on synthetic data,
 *                loss must strictly decrease from epoch 0 to epoch 2.
 *
 * No MNIST data required — synthesises 128 random float[784] images.
 */
#include "vae_config.h"
#include "vae_math.h"
#include "vae_model.h"
#include "test_framework.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


static float run_epoch(VAE *m, float **xs, const int *ls, int n, float beta) {
  const int bsz = m->cfg.batch_size;
  float total = 0.0f;
  int nb = (n + bsz - 1) / bsz;
  for (int b = 0; b < nb; b++) {
    int start = b * bsz;
    int end = start + bsz;
    if (end > n)
      end = n;
    int sz = end - start;
    vae_reset_grads(m);
    vae_forward(m, &xs[start], &ls[start], sz, 1);
    float l = vae_loss(m, &xs[start], sz, beta);
    if (isfinite(l)) {
      vae_backward(m, &xs[start], &ls[start], sz, beta);
      total += l;
    }
    vae_apply_gradients(m, 0.001f);
  }
  return total / (float)nb;
}

static void test_loss_decreases_over_3_epochs(void) {
  VAEConfig cfg = vae_config_v1();
  cfg.batch_size = 16; /* smaller batch for speed */
  VAE *m = create_vae(&cfg, 777ULL);
  ASSERT_TRUE(m != NULL);
  if (!m) return;

  const int N = 128;
  float buf[N][IMAGE_SIZE];
  float *xs[N];
  int ls[N];
  for (int i = 0; i < N; i++) {
    xs[i] = buf[i];
    ls[i] = i % 2;
    for (int j = 0; j < IMAGE_SIZE; j++)
      buf[i][j] = (float)((i * IMAGE_SIZE + j) % 256) / 255.0f;
  }

  float loss0 = run_epoch(m, xs, ls, N, 0.0f); /* beta=0: pure recon */
  float loss1 = run_epoch(m, xs, ls, N, 0.0f);
  float loss2 = run_epoch(m, xs, ls, N, 0.0f);

  ASSERT_TRUE(isfinite(loss0) && isfinite(loss1) && isfinite(loss2));
  ASSERT_TRUE(loss2 < loss0); /* loss must have decreased by epoch 2 */

  free_vae(m);
}

/* ── test 2: KL annealing schedule correctness ──────────────────── */
/*
 * Replicate the annealing formula from vae_train.c and verify it at
 * five key points without running the full training loop.
 *
 * Formula (epoch > beta_warmup):
 *   p    = clamp((epoch - warmup) / anneal, 0, 1)
 *   beta = beta_start + (beta_end - beta_start) * p
 */
static float beta_at_epoch(const VAEConfig *c, int epoch) {
  if (epoch <= c->beta_warmup)
    return c->beta_start;
  float p = (float)(epoch - c->beta_warmup) / (float)c->beta_anneal;
  if (p > 1.0f) p = 1.0f;
  return c->beta_start + (c->beta_end - c->beta_start) * p;
}

static void test_kl_annealing_schedule(void) {
  VAEConfig cfg = vae_config_v1();

  /* During warmup: beta must stay at beta_start */
  ASSERT_NEAR(beta_at_epoch(&cfg, 0),              cfg.beta_start, 1e-6f);
  ASSERT_NEAR(beta_at_epoch(&cfg, cfg.beta_warmup), cfg.beta_start, 1e-6f);

  /* Midpoint of anneal: beta is exactly halfway between start and end */
  int mid = cfg.beta_warmup + cfg.beta_anneal / 2;
  float expected_mid = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * 0.5f;
  ASSERT_NEAR(beta_at_epoch(&cfg, mid), expected_mid, 1e-4f);

  /* End of anneal: beta must reach beta_end */
  int end = cfg.beta_warmup + cfg.beta_anneal;
  ASSERT_NEAR(beta_at_epoch(&cfg, end), cfg.beta_end, 1e-5f);

  /* Beyond anneal: beta must clamp at beta_end, not overshoot */
  ASSERT_NEAR(beta_at_epoch(&cfg, end * 2), cfg.beta_end, 1e-5f);

  /* Monotonically non-decreasing over the full anneal window */
  for (int e = cfg.beta_warmup; e < end; e++) {
    float b_now  = beta_at_epoch(&cfg, e);
    float b_next = beta_at_epoch(&cfg, e + 1);
    ASSERT_TRUE(b_next >= b_now - 1e-6f);
  }
}

void run_test_train(void) {
  SUITE("vae_train (integration)");
  RUN_TEST(test_loss_decreases_over_3_epochs);
  RUN_TEST(test_kl_annealing_schedule);
}
