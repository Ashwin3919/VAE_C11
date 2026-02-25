/*
 * mnist_loader.h — MNIST binary file reader interface
 *
 * With    -DFULL_MNIST : loads all digits 0-9  (60,000 training samples)
 * Without -DFULL_MNIST : loads digits 0-1 only (~12,665 training samples)
 */

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

/*
 * load_mnist_data — read images and labels from MNIST idx files.
 *
 *   img_path : path to train-images-idx3-ubyte (or t10k equivalent)
 *   lbl_path : path to train-labels-idx1-ubyte (or t10k equivalent)
 *   images   : out — array of float[784] pixel buffers, normalised to [0,1]
 *   labels   : out — array of integer class labels
 *   count    : out — number of valid samples loaded
 *
 * Returns 1 on success, 0 on failure.
 * Caller must free each images[i], then images[], then labels[].
 */
int load_mnist_data(const char *img_path, const char *lbl_path,
                    float ***images, int **labels, int *count);

#endif /* MNIST_LOADER_H */
