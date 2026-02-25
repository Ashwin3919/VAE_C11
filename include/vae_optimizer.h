/*
 * vae_optimizer.h — Adam optimiser utility.
 *
 * adam_update() is a stateless helper; the caller holds the moment buffers and
 * the global step counter (VAE.adam_t).  This matches the original semantics
 * exactly while keeping the optimiser logic in one place.
 */
#ifndef VAE_OPTIMIZER_H
#define VAE_OPTIMIZER_H

/*
 * adam_update — update `w` using Adam with the accumulated gradient `dw`.
 *
 *   w    : parameter array (updated in-place)
 *   dw   : gradient array  (read-only; caller resets via vae_reset_grads)
 *   m, v : first / second moment buffers (caller owns, initialised to 0)
 *   n    : number of elements
 *   lr   : learning rate
 *   t    : global step counter (1-indexed; used for bias correction)
 */
void adam_update(float *w, float *dw, float *m, float *v, int n, float lr,
                 int t);

#endif /* VAE_OPTIMIZER_H */
