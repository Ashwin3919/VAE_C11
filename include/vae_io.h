/*
 * vae_io.h — model checkpoint save / load and filesystem helpers.
 */
#ifndef VAE_IO_H
#define VAE_IO_H

#include "vae_model.h"

/*
 * save_model — write weights + Adam state to a binary checkpoint.
 *   Returns 1 on success, 0 on any I/O error.
 */
int save_model(const VAE *m, const char *path);

/*
 * load_model — restore weights + Adam state from a checkpoint.
 *   Returns 1 on success, 0 if file is absent or has the wrong version.
 *   Precondition: m must already be created with create_vae() using a
 *   VAEConfig that matches the one used when save_model() was called
 *   (same h1, h2, latent, num_classes).  Mismatched configs produce
 *   silently incorrect weights; version mismatch is caught by the magic
 *   number and returns 0 with an actionable message.
 */
int load_model(VAE *m, const char *path);

/* Portable recursive mkdir (does not require shell or system()). */
void mkdir_p(const char *path);

#endif /* VAE_IO_H */
