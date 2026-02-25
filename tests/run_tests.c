/*
 * run_tests.c — test runner.  Calls all test suites and exits non-zero on
 * failure.
 *
 * Each test_*.c file defines:
 *   - int _tests_run / _tests_failed  (local to that TU)
 *   - void run_test_<name>(void)
 *
 * This runner accumulates totals across suites.
 */
#include "test_framework.h"
#include <stdio.h>

/* Global counters required by test_framework.h macros */
int _tests_run = 0;
int _tests_failed = 0;

/* Forward declarations */
void run_test_rng(void);
void run_test_optimizer(void);
void run_test_model(void);
void run_test_train(void);

int main(void) {
  printf("Running VAE test suite...\n");

  run_test_rng();
  run_test_optimizer();
  run_test_model();
  run_test_train();

  printf("\n─────────────────────────────────────\n");
  if (_tests_failed == 0) {
    printf("ALL %d TESTS PASSED\n", _tests_run);
    return 0;
  } else {
    printf("FAILED %d / %d TESTS\n", _tests_failed, _tests_run);
    return 1;
  }
}
