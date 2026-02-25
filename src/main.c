/*
 * main.c — entry point with runtime configuration flags.
 *
 * Usage:
 *   ./exe/vae_model                  # v1 model, digits 0-1 (default)
 *   ./exe/vae_model --full-mnist     # v1 model, all 10 digits
 *   ./exe/vae_model --digits 0,1,2   # v1 model, digits 0,1,2
 *
 * Model variant (v1/v2/v3) is still selected at compile time via
 *   -DVERSION_V2 / -DVERSION_V3
 * because it determines static memory layout baked into the slab.
 */
#include "vae_config.h"
#include "vae_generate.h"
#include "vae_io.h"
#include "vae_math.h"
#include "vae_model.h"
#include "vae_train.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* CLI parsing                                                          */
/* ------------------------------------------------------------------ */

/*
 * parse_digits — populate dst[] from a comma-separated string like "0,1,2".
 * Returns the count of parsed digits, or 0 on error.
 * dst must have room for at least 10 ints.
 */
static int parse_digits(const char *arg, int *dst) {
  int n = 0;
  char buf[64];
  strncpy(buf, arg, sizeof buf - 1);
  buf[sizeof buf - 1] = '\0';
  char *tok = strtok(buf, ",");
  while (tok && n < 10) {
    char *end;
    long v = strtol(tok, &end, 10);
    if (end == tok || v < 0 || v > 9) {
      fprintf(stderr, "[ERROR] invalid digit '%s' in --digits\n", tok);
      return 0;
    }
    dst[n++] = (int)v;
    tok = strtok(NULL, ",");
  }
  return n;
}

int main(int argc, char *argv[]) {
#if defined(VERSION_V3)
  VAEConfig cfg = vae_config_v3();
#elif defined(VERSION_V2)
  VAEConfig cfg = vae_config_v2();
#else
  VAEConfig cfg = vae_config_v1();
#endif

  /* ── runtime digit filter ─────────────────────────────────────────── */
  int allowed_digits[10];
  int n_allowed = 0; /* 0 = use config default */

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--full-mnist") == 0) {
      cfg.full_mnist = 1;
      n_allowed = 0; /* NULL filter → all digits */
    } else if (strcmp(argv[i], "--digits") == 0 && i + 1 < argc) {
      n_allowed = parse_digits(argv[++i], allowed_digits);
      if (n_allowed == 0)
        return 1;
      cfg.full_mnist = 0;
    } else {
      fprintf(stderr, "[WARN] unknown argument: %s\n", argv[i]);
    }
  }

  /* If no CLI override, use config default (v1/v2 → binary, v3 → full) */
  if (n_allowed == 0 && !cfg.full_mnist) {
    /* binary default for v1/v2: keep only digits 0 and 1 */
    if (cfg.num_classes == 2) {
      allowed_digits[0] = 0;
      allowed_digits[1] = 1;
      n_allowed = 2;
    }
  }

  printf(
      "CVAE %s  enc=[%d]->%d->%d->[%dz]  dec=[%dz]->%d->%d->[%d]  classes=%d\n",
      cfg.version_tag, cfg.enc_in, cfg.h1, cfg.h2, cfg.latent, cfg.dec_in,
      cfg.h1, cfg.h2, IMAGE_SIZE, cfg.num_classes);

  VAE *m = create_vae(&cfg, (uint64_t)time(NULL));
  if (!m) {
    fprintf(stderr, "[FATAL] out of memory — cannot create model\n");
    return 1;
  }
  load_model(m, cfg.model_file); /* resume if checkpoint exists */

  Dataset *ds =
      load_dataset(&cfg, n_allowed ? allowed_digits : NULL, n_allowed);
  if (!ds) {
    free_vae(m);
    return 1;
  }

  train(m, ds);

  printf("\n[GEN] final samples per digit...\n");
  for (int cls = 0; cls < cfg.num_classes; cls++)
    for (int s = 0; s < 3; s++)
      generate_digit(m, cls, 0.8f, 9999, s);
  printf("[GEN] done -> %s/epoch_9999_digit*_s*.pgm\n", cfg.result_dir);

  free_dataset(ds);
  free_vae(m);
  return 0;
}
