/*
 * vae_generate.h — image generation from the trained decoder.
 */
#ifndef VAE_GENERATE_H
#define VAE_GENERATE_H

#include "vae_model.h"

/*
 * generate_digit — write one PGM file by sampling z ~ N(0, temp²I)
 *                  and running it through the decoder conditioned on label.
 */
void generate_digit(VAE *m, int label, float temp, int epoch, int sample_idx);

/* Generate 3 samples per digit class and print a summary line. */
void generate_epoch_samples(VAE *m, int epoch);

#endif /* VAE_GENERATE_H */
