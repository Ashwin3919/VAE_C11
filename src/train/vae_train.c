/*
 * vae_train.c — training loop with LR schedule, KL annealing, batched
 * validation, early stopping, and MNIST dataset management.
 */
#include "vae_train.h"
#include "mnist_loader.h"
#include "vae_generate.h"
#include "vae_io.h"
#include "vae_math.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif


/* ------------------------------------------------------------------ */
/* Dataset                                                              */
/* ------------------------------------------------------------------ */

Dataset *load_dataset(const VAEConfig *cfg, const int *allowed_digits,
                      int n_allowed) {
  Dataset *ds = malloc(sizeof(Dataset));
  if (!ds)
    return NULL;
  char train_img[PATH_BUF_SIZE], train_lbl[PATH_BUF_SIZE];
  snprintf(train_img, sizeof train_img, "%s/train-images-idx3-ubyte",
           cfg->data_dir);
  snprintf(train_lbl, sizeof train_lbl, "%s/train-labels-idx1-ubyte",
           cfg->data_dir);
  if (load_mnist_data(train_img, train_lbl, &ds->images, &ds->labels,
                      &ds->count, allowed_digits, n_allowed)) {
    printf("[INFO] loaded %d images\n", ds->count);
    return ds;
  }
  fprintf(stderr, "[ERROR] MNIST data not found — run ./download_mnist.sh\n");
  free(ds);
  return NULL;
}

void free_dataset(Dataset *ds) {
  if (!ds)
    return;
  for (int i = 0; i < ds->count; i++)
    free(ds->images[i]);
  free(ds->images);
  free(ds->labels);
  free(ds);
}

/* ------------------------------------------------------------------ */
/* Fisher-Yates shuffle of [lo, hi) range                              */
/* ------------------------------------------------------------------ */

static void shuffle_range(Dataset *ds, Rng *rng, int lo, int hi) {
  for (int i = hi - 1; i > lo; i--) {
    int j = lo + rng_int(rng, i - lo + 1);
    float *ti = ds->images[i];
    ds->images[i] = ds->images[j];
    ds->images[j] = ti;
    int tl = ds->labels[i];
    ds->labels[i] = ds->labels[j];
    ds->labels[j] = tl;
  }
}

/* ------------------------------------------------------------------ */
/* Training loop                                                        */
/* ------------------------------------------------------------------ */

void train(VAE *m, Dataset *ds) {
  const VAEConfig *c = &m->cfg;
  mkdir_p(c->model_dir);
  mkdir_p(c->result_dir);

  /* One-time shuffle then split: 90% train / 10% val */
  shuffle_range(ds, &m->rng, 0, ds->count);
  int val_count = ds->count / 10;
  int train_count = ds->count - val_count;
  printf(
      "[TRAIN] %d train  %d val  |  %d epochs  lr=%.4f  batch=%d  classes=%d\n",
      train_count, val_count, c->epochs, (double)c->lr, c->batch_size,
      c->num_classes);

  float best_val = 1e9f;
  int patience = 0;

  for (int epoch = 0; epoch < c->epochs; epoch++) {
#ifdef _OPENMP
    double t0 = omp_get_wtime();
#else
    clock_t t0 = clock();
#endif

    /* Learning rate: linear warmup then cosine decay, floor 5% */
    float lr;
    if (epoch < c->lr_warmup_epochs) {
      lr = c->lr * (float)(epoch + 1) / (float)c->lr_warmup_epochs;
    } else {
      float p = (float)(epoch - c->lr_warmup_epochs) /
                (float)(c->epochs - c->lr_warmup_epochs);
      lr = c->lr * 0.5f * (1.0f + cosf(M_PI_F * p));
      if (lr < c->lr * 0.05f)
        lr = c->lr * 0.05f;
    }

    /* KL annealing */
    float beta = c->beta_start;
    if (epoch > c->beta_warmup) {
      float p = (float)(epoch - c->beta_warmup) / (float)c->beta_anneal;
      if (p > 1.0f)
        p = 1.0f;
      beta = c->beta_start + (c->beta_end - c->beta_start) * p;
    }

    /* Shuffle training partition */
    shuffle_range(ds, &m->rng, 0, train_count);

    /* ── training batches ── */
    float total_loss = 0.0f;
    int nb = (train_count + c->batch_size - 1) / c->batch_size;

    for (int b = 0; b < nb; b++) {
      int start = b * c->batch_size;
      int end = start + c->batch_size;
      if (end > train_count)
        end = train_count;
      int bsz = end - start;

      vae_reset_grads(m);
      vae_forward(m, &ds->images[start], &ds->labels[start], bsz, 1);
      float bloss = vae_loss(m, &ds->images[start], bsz, beta);
      if (isfinite(bloss)) {
        /*
         * Normal batch: accumulate gradients and take an Adam step.
         * vae_model.h invariant: "vae_apply_gradients() MUST NOT be called
         * if vae_backward() was skipped."  Both calls are guarded together
         * so the invariant cannot be violated even if this block is refactored.
         */
        vae_backward(m, &ds->images[start], &ds->labels[start], bsz, beta);
        vae_apply_gradients(m, lr);
      } else {
        /*
         * NaN/Inf batch — skip backward AND apply_gradients entirely.
         * Calling apply_gradients without a preceding backward would advance
         * adam_t and apply stale first/second moments from prior batches,
         * producing a silent phantom update.  Substitute a sentinel loss of
         * 1.0 so the epoch average reflects the bad batch without corrupting
         * the optimiser state.
         */
        bloss = 1.0f;
      }
      total_loss += bloss;
    }
    float train_avg = total_loss / (float)nb;

    /* ── batched validation ── */
    float val_loss = 0.0f;
    int val_ok = 0;
    for (int i = train_count; i < ds->count; i += c->batch_size) {
      int bsz = c->batch_size;
      if (i + bsz > ds->count)
        bsz = ds->count - i;
      vae_forward(m, &ds->images[i], &ds->labels[i], bsz, 0);
      float l = vae_loss(m, &ds->images[i], bsz, beta);
      if (isfinite(l)) {
        val_loss += l * bsz;
        val_ok += bsz;
      }
    }
    val_loss = val_ok > 0 ? val_loss / (float)val_ok : 1e9f;

    /* ── early stopping on val loss ── */
    if (val_loss < best_val) {
      best_val = val_loss;
      patience = 0;
      save_model(m, c->model_file);
    } else {
      patience++;
      if (patience > c->es_patience && epoch > c->es_min_epoch) {
        printf("[TRAIN] early stop epoch %d  best_val=%.4f\n", epoch,
               (double)best_val);
        break;
      }
    }

#ifdef _OPENMP
    double dt = omp_get_wtime() - t0;
#else
    double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
#endif
    double sps = dt > 0 ? (double)train_count / dt : 0;

    if (epoch % 10 == 0 || epoch < 10)
      printf("[EPOCH %3d] train=%.4f  val=%.4f  best=%.4f"
             "  beta=%.5f  lr=%.5f  %.0f img/s\n",
             epoch, (double)train_avg, (double)val_loss, (double)best_val,
             (double)beta, (double)lr, sps);

    if (epoch % c->save_every == 0)
      generate_epoch_samples(m, epoch);
  }
  printf("[TRAIN] done  best_val=%.4f  model=%s\n", (double)best_val,
         c->model_file);
}
