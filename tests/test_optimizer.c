/*
 * test_optimizer.c — unit tests for Adam.
 *
 * Tests that Adam successfully minimises f(x) = x² from x=10 to |x|<0.1
 * within 500 steps (it should converge in <100 for this trivial case).
 */
#include "vae_optimizer.h"
#include "test_framework.h"

#include <math.h>
#include <string.h>


static void test_adam_convergence_scalar(void) {
  float w = 10.0f, dw = 0.0f, m = 0.0f, v = 0.0f;
  for (int t = 1; t <= 500; t++) {
    dw = 2.0f * w; /* gradient of x² */
    adam_update(&w, &dw, &m, &v, 1, 0.1f, t);
  }
  ASSERT_TRUE(fabsf(w) < 0.1f);
}

/*
 * Gradient zeroing is the sole responsibility of vae_reset_grads(), which
 * calls memset before each batch.  adam_update() intentionally does NOT zero
 * dw — keeping the zeroing in one place eliminates a hidden side effect.
 * This test verifies the new contract: dw is unchanged after an update.
 */
static void test_adam_does_not_zero_gradient(void) {
  float w = 1.0f, dw = 5.0f, m = 0.0f, v = 0.0f;
  adam_update(&w, &dw, &m, &v, 1, 0.01f, 1);
  ASSERT_NEAR(dw, 5.0f, 1e-9f); /* dw must be unchanged — caller resets it */
}

static void test_adam_vector_convergence(void) {
  const int N = 8;
  float w[N], dw[N], m[N], v[N];
  memset(m, 0, sizeof m);
  memset(v, 0, sizeof v);
  for (int i = 0; i < N; i++)
    w[i] = (float)(i + 1);

  for (int t = 1; t <= 1000; t++) {
    for (int i = 0; i < N; i++)
      dw[i] = 2.0f * w[i];
    adam_update(w, dw, m, v, N, 0.1f, t);
  }
  for (int i = 0; i < N; i++)
    ASSERT_TRUE(fabsf(w[i]) < 0.1f);
}

void run_test_optimizer(void) {
  SUITE("vae_optimizer");
  RUN_TEST(test_adam_convergence_scalar);
  RUN_TEST(test_adam_does_not_zero_gradient);
  RUN_TEST(test_adam_vector_convergence);
}
