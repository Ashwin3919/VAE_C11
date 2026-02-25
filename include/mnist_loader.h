/*
 * mnist_loader.h — MNIST binary file reader interface
 *
 * Digit filtering is a runtime parameter, not a compile-time flag.
 * Pass allowed_digits=NULL / n_allowed=0 to load all 10 classes.
 * Pass e.g. allowed_digits={0,1} / n_allowed=2 for binary-only mode.
 */

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

/*
 * load_mnist_data — read images and labels from MNIST idx files.
 *
 *   img_path       : path to train-images-idx3-ubyte (or t10k equivalent)
 *   lbl_path       : path to train-labels-idx1-ubyte (or t10k equivalent)
 *   images         : out — array of float[784] pixel buffers, normalised [0,1]
 *   labels         : out — array of integer class labels
 *   count          : out — number of valid samples loaded
 *   allowed_digits : array of digit values to keep (e.g. {0,1}), or NULL
 *   n_allowed      : length of allowed_digits; 0 = load all digits
 *
 * Returns 1 on success, 0 on failure.
 * Caller must free each images[i], then images[], then labels[].
 */
int load_mnist_data(const char *img_path, const char *lbl_path, float ***images,
                    int **labels, int *count, const int *allowed_digits,
                    int n_allowed);

#endif /* MNIST_LOADER_H */
