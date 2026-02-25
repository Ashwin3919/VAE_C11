/*
 * vae_train.h — training loop and dataset management.
 */
#ifndef VAE_TRAIN_H
#define VAE_TRAIN_H

#include "vae_model.h"

typedef struct {
  float **images;
  int *labels;
  int count;
} Dataset;

/*
 * load_dataset — load MNIST from cfg->data_dir.
 *   Respects cfg->full_mnist: 0 = digits 0-1 only; 1 = all 10 digits.
 *   Returns a heap-allocated Dataset on success, NULL on I/O error or OOM.
 *   Caller must release with free_dataset().
 */
Dataset *load_dataset(const VAEConfig *cfg);

/*
 * free_dataset — release all memory owned by ds.
 *   Safe to call with NULL.
 */
void free_dataset(Dataset *ds);

/*
 * train — run the full training loop on ds.
 *   Handles: Fisher-Yates shuffle, 90/10 train/val split, LR warmup +
 *   cosine decay, KL annealing, batched validation, early stopping, and
 *   periodic checkpoint saves (every cfg->save_every epochs).
 *
 *   Precondition: m was created with create_vae(&cfg, seed) where cfg
 *   matches ds (num_classes must be consistent with labels in ds).
 *   Postcondition: m->_mem contains the weights from the best validation
 *   epoch; the checkpoint file at cfg->model_file holds the same weights.
 */
void train(VAE *m, Dataset *ds);

#endif /* VAE_TRAIN_H */
