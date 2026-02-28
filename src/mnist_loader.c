/*
 * mnist_loader.c — reads binary MNIST idx files.
 *
 * Digit filtering is controlled at runtime via the allowed_digits/n_allowed
 * parameters — no compile-time #defines required.  Pass NULL/0 to load all
 * 10 classes; pass e.g. {0,1}/2 for the binary subset.
 */

#include "mnist_loader.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MNIST_IMG_MAGIC 0x00000803u
#define MNIST_LBL_MAGIC 0x00000801u

static uint32_t bswap32(uint32_t x) {
  return ((x & 0xFFu) << 24) | (((x >> 8) & 0xFFu) << 16) |
         (((x >> 16) & 0xFFu) << 8) | ((x >> 24) & 0xFFu);
}

/* Read one big-endian uint32; returns 0 on I/O error. */
static int read_be32(FILE *f, uint32_t *out) {
  uint32_t v;
  if (fread(&v, sizeof v, 1, f) != 1)
    return 0;
  *out = bswap32(v);
  return 1;
}

/* Returns 1 if digit lbl is in the allowed set (or if n_allowed == 0). */
static int digit_allowed(unsigned char lbl, const int *allowed, int n_allowed) {
  if (n_allowed == 0)
    return 1;
  for (int i = 0; i < n_allowed; i++)
    if ((int)lbl == allowed[i])
      return 1;
  return 0;
}

int load_mnist_data(const char *img_path, const char *lbl_path, float ***images,
                    int **labels, int *count, const int *allowed_digits,
                    int n_allowed) {
  FILE *fi = fopen(img_path, "rb");
  FILE *fl = fopen(lbl_path, "rb");

  if (!fi || !fl) {
    fprintf(stderr, "[ERROR] cannot open MNIST files:\n  %s\n  %s\n", img_path,
            lbl_path);
    fprintf(stderr, "        run ./download_mnist.sh to fetch them.\n");
    if (fi)
      fclose(fi);
    if (fl)
      fclose(fl);
    return 0;
  }

  /* --- image header --- */
  uint32_t magic, n_imgs, rows, cols;
  if (!read_be32(fi, &magic) || !read_be32(fi, &n_imgs) ||
      !read_be32(fi, &rows) || !read_be32(fi, &cols)) {
    fprintf(stderr, "[ERROR] truncated image header: %s\n", img_path);
    fclose(fi);
    fclose(fl);
    return 0;
  }
  if (magic != MNIST_IMG_MAGIC || rows != 28 || cols != 28) {
    fprintf(stderr, "[ERROR] bad image header (magic=0x%08X rows=%u cols=%u)\n",
            magic, rows, cols);
    fclose(fi);
    fclose(fl);
    return 0;
  }

  /* --- label header --- */
  uint32_t lmagic, n_lbls;
  if (!read_be32(fl, &lmagic) || !read_be32(fl, &n_lbls)) {
    fprintf(stderr, "[ERROR] truncated label header: %s\n", lbl_path);
    fclose(fi);
    fclose(fl);
    return 0;
  }
  if (lmagic != MNIST_LBL_MAGIC || n_lbls != n_imgs) {
    fprintf(stderr, "[ERROR] bad label header or size mismatch (%u vs %u)\n",
            n_lbls, n_imgs);
    fclose(fi);
    fclose(fl);
    return 0;
  }

  /* --- first pass: read all labels, count valid samples --- */
  unsigned char *lbuf = malloc(n_imgs);
  if (!lbuf) {
    fclose(fi);
    fclose(fl);
    return 0;
  }

  if (fread(lbuf, 1, n_imgs, fl) != n_imgs) {
    fprintf(stderr, "[ERROR] short read on labels: %s\n", lbl_path);
    free(lbuf);
    fclose(fi);
    fclose(fl);
    return 0;
  }

  int valid = 0;
  for (uint32_t i = 0; i < n_imgs; i++)
    if (digit_allowed(lbuf[i], allowed_digits, n_allowed))
      valid++;

  /* Print a human-readable summary of the active filter */
  if (n_allowed == 0) {
    printf("[INFO] MNIST: %u images, %d valid (digits 0-9)\n", n_imgs, valid);
  } else {
    printf("[INFO] MNIST: %u images, %d valid (digits", n_imgs, valid);
    for (int i = 0; i < n_allowed; i++)
      printf("%s%d", i == 0 ? " " : ",", allowed_digits[i]);
    printf(")\n");
  }

  *images = malloc((size_t)valid * sizeof(float *));
  *labels = malloc((size_t)valid * sizeof(int));
  *count = valid;
  if (!*images || !*labels) {
    free(*images);
    free(*labels);
    free(lbuf);
    fclose(fi);
    fclose(fl);
    return 0;
  }

  /* --- second pass: load pixel data for valid samples --- */
  unsigned char pbuf[784];
  int idx = 0;
  for (uint32_t i = 0; i < n_imgs; i++) {
    if (fread(pbuf, 1, 784, fi) != 784) {
      fprintf(stderr, "[ERROR] short pixel read at sample %u: %s\n", i,
              img_path);
      for (int k = 0; k < idx; k++)
        free((*images)[k]);
      free(*images);
      free(*labels);
      free(lbuf);
      fclose(fi);
      fclose(fl);
      return 0;
    }
    if (!digit_allowed(lbuf[i], allowed_digits, n_allowed))
      continue;

    (*images)[idx] = malloc(784 * sizeof(float));
    if (!(*images)[idx]) {
      for (int k = 0; k < idx; k++)
        free((*images)[k]);
      free(*images);
      free(*labels);
      free(lbuf);
      fclose(fi);
      fclose(fl);
      return 0;
    }
    (*labels)[idx] = (int)lbuf[i];
    for (int j = 0; j < 784; j++)
      (*images)[idx][j] = (float)pbuf[j] / 255.0f;
    idx++;
  }

  free(lbuf);
  fclose(fi);
  fclose(fl);
  return 1;
}
