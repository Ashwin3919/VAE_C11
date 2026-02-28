/*
 * test_rng.c — unit tests for vae_rng (reproducibility, statistics).
 */
#include "vae_rng.h"
#include "test_framework.h"

#include <math.h>

/* ── counters required by test_framework.h ── */

/* Same seed → same sequence */
static void test_rng_determinism(void) {
  Rng a, b;
  rng_init(&a, 42);
  rng_init(&b, 42);
  for (int i = 0; i < 1000; i++)
    ASSERT_NEAR(rng_uniform(&a), rng_uniform(&b), 0.0f);
}

/* Different seeds → different sequences */
static void test_rng_distinct_seeds(void) {
  Rng a, b;
  rng_init(&a, 1);
  rng_init(&b, 2);
  int same = 0;
  for (int i = 0; i < 100; i++)
    same += (rng_uniform(&a) == rng_uniform(&b));
  ASSERT_TRUE(same < 5); /* extremely unlikely to match more than ~0 */
}

/* Uniform output in [0, 1) */
static void test_rng_uniform_range(void) {
  Rng r;
  rng_init(&r, 99);
  for (int i = 0; i < 10000; i++) {
    float u = rng_uniform(&r);
    ASSERT_TRUE(u >= 0.0f && u < 1.0f);
  }
}

/* Normal samples: |mean| < 0.05, |var - 1| < 0.05  (10k samples) */
static void test_rng_normal_stats(void) {
  Rng r;
  rng_init(&r, 7);
  float sum = 0.0f, sum2 = 0.0f;
  const int N = 10000;
  for (int i = 0; i < N; i++) {
    float x = rng_normal(&r);
    sum += x;
    sum2 += x * x;
  }
  float mean = sum / N;
  float var = sum2 / N - mean * mean;
  ASSERT_NEAR(mean, 0.0f, 0.05f);
  ASSERT_NEAR(var, 1.0f, 0.05f);
}

/* rng_int covers [0, n-1] */
static void test_rng_int_range(void) {
  Rng r;
  rng_init(&r, 123);
  for (int i = 0; i < 10000; i++) {
    int v = rng_int(&r, 10);
    ASSERT_TRUE(v >= 0 && v < 10);
  }
}

void run_test_rng(void) {
  SUITE("vae_rng");
  RUN_TEST(test_rng_determinism);
  RUN_TEST(test_rng_distinct_seeds);
  RUN_TEST(test_rng_uniform_range);
  RUN_TEST(test_rng_normal_stats);
  RUN_TEST(test_rng_int_range);
}
