/*
 * test_framework.h — minimal unit test macros.  No external dependencies.
 */
#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Global pass/fail counters — updated by each assertion. */
extern int _tests_run;
extern int _tests_failed;

#define ASSERT_EQ(a, b)                                                        \
  do {                                                                         \
    _tests_run++;                                                              \
    if ((a) != (b)) {                                                          \
      fprintf(stderr, "  FAIL %s:%d  %s == %s  (%d != %d)\n", __FILE__,        \
              __LINE__, #a, #b, (int)(a), (int)(b));                           \
      _tests_failed++;                                                         \
    }                                                                          \
  } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
  do {                                                                         \
    _tests_run++;                                                              \
    float _a = (a), _b = (b), _t = (tol);                                      \
    if (fabsf(_a - _b) > _t) {                                                 \
      fprintf(stderr, "  FAIL %s:%d  |%s - %s| = %.6f > %.6f\n", __FILE__,     \
              __LINE__, #a, #b, (double)fabsf(_a - _b), (double)_t);           \
      _tests_failed++;                                                         \
    }                                                                          \
  } while (0)

#define ASSERT_TRUE(cond)                                                      \
  do {                                                                         \
    _tests_run++;                                                              \
    if (!(cond)) {                                                             \
      fprintf(stderr, "  FAIL %s:%d  expected true: %s\n", __FILE__, __LINE__, \
              #cond);                                                          \
      _tests_failed++;                                                         \
    }                                                                          \
  } while (0)

#define RUN_TEST(fn)                                                           \
  do {                                                                         \
    printf("  %-40s", #fn);                                                    \
    fflush(stdout);                                                            \
    int _before = _tests_failed;                                               \
    fn();                                                                      \
    printf("%s\n", _tests_failed == _before ? "OK" : "FAILED");                \
  } while (0)

#define SUITE(name) printf("\n=== %s ===\n", name)

#endif /* TEST_FRAMEWORK_H */
