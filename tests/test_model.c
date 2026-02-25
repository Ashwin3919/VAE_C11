/*
 * test_model.c — unit tests for the VAE model.
 *
 * Tests (no MNIST data required):
 *   1. Forward pass determinism  — same seed → identical output
 *   2. Loss is positive and finite on random input
 *   3. Checkpoint round-trip — save then load, forward matches
 *   4. Backward produces non-zero gradients on a single batch
 *   5. Numerical gradient check — finite-difference vs analytical
 */
#include "vae_config.h"
#include "vae_io.h"
#include "vae_math.h"
#include "vae_model.h"
#include "test_framework.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* Create a tiny v1 model with fixed seed for reproducibility. */
static VAE *make_test_model(void) {
  VAEConfig cfg = vae_config_v1();
  return create_vae(&cfg, /*seed=*/12345ULL);
}

/* ── test 1: determinism ─────────────────────────────────────────── */
static void test_forward_determinism(void) {
  /* Two models with the same seed must produce identical output. */
  VAE *a = make_test_model();
  VAE *b = make_test_model();
  ASSERT_TRUE(a != NULL);
  ASSERT_TRUE(b != NULL);
  if (!a || !b) { free_vae(a); free_vae(b); return; }

  float img[IMAGE_SIZE];
  for (int j = 0; j < IMAGE_SIZE; j++)
    img[j] = 0.5f;
  float *xs[1] = {img};
  int ls[1] = {0};

  vae_forward(a, xs, ls, 1, /*training=*/1);
  vae_forward(b, xs, ls, 1, /*training=*/1);

  /* Same seed → same weights and same RNG draws → bit-identical output.
   * Use 1e-6f rather than 0.0f: -ffast-math may reorder float ops between
   * translation units, causing sub-ULP differences. */
  for (int i = 0; i < IMAGE_SIZE; i++)
    ASSERT_NEAR(a->output[i], b->output[i], 1e-6f);

  free_vae(a);
  free_vae(b);
}

/* ── test 2: loss is positive and finite ─────────────────────────── */
static void test_loss_positive_finite(void) {
  VAE *m = make_test_model();
  ASSERT_TRUE(m != NULL);
  if (!m) return;
  const int bsz = 4;

  float buf[bsz][IMAGE_SIZE];
  float *xs[bsz];
  int ls[bsz];
  for (int i = 0; i < bsz; i++) {
    xs[i] = buf[i];
    ls[i] = i % 2;
    srand((unsigned)i * 31 + 7);
    for (int j = 0; j < IMAGE_SIZE; j++)
      xs[i][j] = (float)rand() / (float)RAND_MAX;
  }

  vae_forward(m, xs, ls, bsz, 1);
  float loss = vae_loss(m, xs, bsz, 0.0001f);

  ASSERT_TRUE(isfinite(loss));
  ASSERT_TRUE(loss > 0.0f);

  free_vae(m);
}

/* ── test 3: checkpoint round-trip ───────────────────────────────── */
static void test_checkpoint_roundtrip(void) {
  const char *path = "/tmp/vae_test_ckpt.bin";
  VAE *m = make_test_model();
  ASSERT_TRUE(m != NULL);
  if (!m) return;

  float img[IMAGE_SIZE];
  for (int j = 0; j < IMAGE_SIZE; j++)
    img[j] = (float)j / IMAGE_SIZE;
  float *xs[1] = {img};
  int ls[1] = {1};

  vae_forward(m, xs, ls, 1, 0);
  float out_before[IMAGE_SIZE];
  memcpy(out_before, m->output, IMAGE_SIZE * sizeof(float));

  ASSERT_TRUE(save_model(m, path));

  VAEConfig cfg = vae_config_v1();
  VAE *m2 = create_vae(&cfg, 99999ULL); /* different seed */
  ASSERT_TRUE(m2 != NULL);
  if (!m2) { free_vae(m); remove(path); return; }
  ASSERT_TRUE(load_model(m2, path));

  vae_forward(m2, xs, ls, 1, 0);
  for (int i = 0; i < IMAGE_SIZE; i++)
    ASSERT_NEAR(m2->output[i], out_before[i], 1e-5f);

  free_vae(m);
  free_vae(m2);
  remove(path);
}

/* ── test 4: backward produces non-zero gradients ────────────────── */
static void test_backward_nonzero_grads(void) {
  VAE *m = make_test_model();
  ASSERT_TRUE(m != NULL);
  if (!m) return;

  float img[IMAGE_SIZE];
  for (int j = 0; j < IMAGE_SIZE; j++)
    img[j] = 0.3f;
  float *xs[1] = {img};
  int ls[1] = {0};

  vae_reset_grads(m);
  vae_forward(m, xs, ls, 1, 1);
  vae_backward(m, xs, ls, 1, 0.0001f);

  /* At least some output-layer bias gradients must be non-zero */
  float sum = 0.0f;
  for (int i = 0; i < IMAGE_SIZE; i++)
    sum += fabsf(m->db3[i]);
  ASSERT_TRUE(sum > 1e-10f);

  free_vae(m);
}

/* ── test 5: numerical gradient check ───────────────────────────── */
/*
 * Gold-standard backprop verification: for a handful of output-layer bias
 * parameters, compare the analytical gradient (from vae_backward) to the
 * central-difference numerical gradient:
 *
 *   ∂L/∂θ ≈ (L(θ+ε) − L(θ−ε)) / 2ε
 *
 * training=0 is used so that z = mu (no stochastic sampling), making the
 * forward pass deterministic and the gradient well-defined.
 */
static void test_gradient_check(void) {
  VAE *m = make_test_model();
  ASSERT_TRUE(m != NULL);
  if (!m) return;

  float img[IMAGE_SIZE];
  for (int j = 0; j < IMAGE_SIZE; j++)
    img[j] = (float)j / IMAGE_SIZE; /* smooth ramp, avoids saturation */
  float *xs[1] = {img};
  int ls[1] = {0};

  const float eps  = 1e-3f;
  const float tol  = 1e-2f; /* generous: float32 finite-diff accuracy */
  const float beta = 0.0001f;

  /* Analytical gradient at the operating point */
  vae_reset_grads(m);
  vae_forward(m, xs, ls, 1, /*training=*/0);
  vae_backward(m, xs, ls, 1, beta);

  /* Check 4 representative output-layer biases (dec_b3).
   * b3 only affects the corresponding output pixel, so the chain is
   * b3[k] → pre_out[k] → output[k] → recon_loss — clean and easy to
   * verify numerically. */
  for (int k = 0; k < 4; k++) {
    float analytical = m->db3[k]; /* db3 = ∂L/∂b3, scaled by 1/bsz */

    float orig = m->dec_b3[k];

    m->dec_b3[k] = orig + eps;
    vae_forward(m, xs, ls, 1, 0);
    float L_plus = vae_loss(m, xs, 1, beta);

    m->dec_b3[k] = orig - eps;
    vae_forward(m, xs, ls, 1, 0);
    float L_minus = vae_loss(m, xs, 1, beta);

    m->dec_b3[k] = orig; /* restore */

    float numerical = (L_plus - L_minus) / (2.0f * eps);
    ASSERT_NEAR(analytical, numerical, tol);
  }

  free_vae(m);
}

/* ── test 6: single-sample batch (bsz=1) ────────────────────────── */
/*
 * bsz=1 exercises the per-sample pointer arithmetic (ptr + 0*stride) and
 * the single-sample backward scratch buffers.  Off-by-one errors in loop
 * bounds or stride calculations surface as wrong outputs or UBSan hits.
 */
static void test_forward_bsz1(void) {
  VAE *m = make_test_model();
  ASSERT_TRUE(m != NULL);
  if (!m) return;

  float img[IMAGE_SIZE];
  for (int j = 0; j < IMAGE_SIZE; j++)
    img[j] = (float)j / IMAGE_SIZE;
  float *xs[1] = {img};
  int ls[1] = {0};

  vae_reset_grads(m);
  vae_forward(m, xs, ls, /*bsz=*/1, /*training=*/1);
  float loss = vae_loss(m, xs, /*bsz=*/1, 0.001f);
  ASSERT_TRUE(isfinite(loss));
  ASSERT_TRUE(loss > 0.0f);

  vae_backward(m, xs, ls, /*bsz=*/1, 0.001f);

  /* Output-layer bias gradients must be non-zero for bsz=1 */
  float sum = 0.0f;
  for (int i = 0; i < IMAGE_SIZE; i++)
    sum += fabsf(m->db3[i]);
  ASSERT_TRUE(sum > 1e-10f);

  free_vae(m);
}

/* ── test 7: extreme pixel values (0.0 and 1.0) ─────────────────── */
/*
 * Pixels at exactly 0.0 or 1.0 are valid MNIST values but would produce
 * log(0) in the BCE loss without the clamp guard:
 *   p = vae_clamp(out[i], 1e-7f, 1.0f - 1e-7f)
 * This test verifies that the clamp prevents -Inf loss on saturated inputs.
 */
static void test_extreme_pixel_values(void) {
  VAE *m = make_test_model();
  ASSERT_TRUE(m != NULL);
  if (!m) return;

  /* All-zeros input */
  float img_zeros[IMAGE_SIZE];
  memset(img_zeros, 0, sizeof img_zeros);
  float *xs0[1] = {img_zeros};
  int ls0[1] = {0};

  vae_forward(m, xs0, ls0, 1, /*training=*/0);
  float loss0 = vae_loss(m, xs0, 1, 0.001f);
  ASSERT_TRUE(isfinite(loss0));
  ASSERT_TRUE(loss0 > 0.0f);

  /* All-ones input */
  float img_ones[IMAGE_SIZE];
  for (int j = 0; j < IMAGE_SIZE; j++) img_ones[j] = 1.0f;
  float *xs1[1] = {img_ones};
  int ls1[1] = {1};

  vae_forward(m, xs1, ls1, 1, /*training=*/0);
  float loss1 = vae_loss(m, xs1, 1, 0.001f);
  ASSERT_TRUE(isfinite(loss1));
  ASSERT_TRUE(loss1 > 0.0f);

  /* Alternating 0/1 checkerboard — exercises both branches per batch */
  float img_alt[IMAGE_SIZE];
  for (int j = 0; j < IMAGE_SIZE; j++) img_alt[j] = (float)(j % 2);
  float *xsa[1] = {img_alt};
  int lsa[1] = {0};

  vae_forward(m, xsa, lsa, 1, /*training=*/0);
  float loss_alt = vae_loss(m, xsa, 1, 0.001f);
  ASSERT_TRUE(isfinite(loss_alt));

  free_vae(m);
}

void run_test_model(void) {
  SUITE("vae_model");
  RUN_TEST(test_forward_determinism);
  RUN_TEST(test_loss_positive_finite);
  RUN_TEST(test_checkpoint_roundtrip);
  RUN_TEST(test_backward_nonzero_grads);
  RUN_TEST(test_gradient_check);
  RUN_TEST(test_forward_bsz1);
  RUN_TEST(test_extreme_pixel_values);
}
