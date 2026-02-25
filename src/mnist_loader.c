/*
 * mnist_loader.c — reads binary MNIST idx files
 *
 * Supports binary mode (digits 0-1) and full MNIST (0-9).
 * Compile with -DFULL_MNIST to load all 10 classes.
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
  if (fread(&v, sizeof v, 1, f) != 1) return 0;
  *out = bswap32(v);
  return 1;
}

int load_mnist_data(const char *img_path, const char *lbl_path,
                    float ***images, int **labels, int *count) {
  FILE *fi = fopen(img_path, "rb");
  FILE *fl = fopen(lbl_path, "rb");

  if (!fi || !fl) {
    fprintf(stderr, "[ERROR] cannot open MNIST files:\n  %s\n  %s\n",
            img_path, lbl_path);
    fprintf(stderr, "        run ./download_mnist.sh to fetch them.\n");
    if (fi) fclose(fi);
    if (fl) fclose(fl);
    return 0;
  }

  /* --- image header --- */
  uint32_t magic, n_imgs, rows, cols;
  if (!read_be32(fi, &magic) || !read_be32(fi, &n_imgs) ||
      !read_be32(fi, &rows)  || !read_be32(fi, &cols)) {
    fprintf(stderr, "[ERROR] truncated image header: %s\n", img_path);
    fclose(fi); fclose(fl); return 0;
  }
  if (magic != MNIST_IMG_MAGIC || rows != 28 || cols != 28) {
    fprintf(stderr, "[ERROR] bad image header (magic=0x%08X rows=%u cols=%u)\n",
            magic, rows, cols);
    fclose(fi); fclose(fl); return 0;
  }

  /* --- label header --- */
  uint32_t lmagic, n_lbls;
  if (!read_be32(fl, &lmagic) || !read_be32(fl, &n_lbls)) {
    fprintf(stderr, "[ERROR] truncated label header: %s\n", lbl_path);
    fclose(fi); fclose(fl); return 0;
  }
  if (lmagic != MNIST_LBL_MAGIC || n_lbls != n_imgs) {
    fprintf(stderr, "[ERROR] bad label header or size mismatch (%u vs %u)\n",
            n_lbls, n_imgs);
    fclose(fi); fclose(fl); return 0;
  }

  /* --- first pass: read all labels, count valid samples --- */
  unsigned char *lbuf = malloc(n_imgs);
  if (!lbuf) { fclose(fi); fclose(fl); return 0; }

  if (fread(lbuf, 1, n_imgs, fl) != n_imgs) {
    fprintf(stderr, "[ERROR] short read on labels: %s\n", lbl_path);
    free(lbuf); fclose(fi); fclose(fl); return 0;
  }

  int valid = 0;
  for (uint32_t i = 0; i < n_imgs; i++) {
#ifdef FULL_MNIST
    if (lbuf[i] <= 9) valid++;
#else
    if (lbuf[i] == 0 || lbuf[i] == 1) valid++;
#endif
  }

#ifdef FULL_MNIST
  printf("[INFO] MNIST: %u images, %d valid (digits 0-9)\n", n_imgs, valid);
#else
  printf("[INFO] MNIST: %u images, %d valid (digits 0-1)\n", n_imgs, valid);
#endif

  *images = malloc(valid * sizeof(float *));
  *labels = malloc(valid * sizeof(int));
  *count  = valid;
  if (!*images || !*labels) {
    free(*images); free(*labels);
    free(lbuf); fclose(fi); fclose(fl); return 0;
  }

  /* --- second pass: load pixel data for valid samples --- */
  unsigned char pbuf[784];
  int idx = 0;
  for (uint32_t i = 0; i < n_imgs; i++) {
    if (fread(pbuf, 1, 784, fi) != 784) {
      fprintf(stderr, "[ERROR] short pixel read at sample %u: %s\n",
              i, img_path);
      /* clean up already-allocated rows */
      for (int k = 0; k < idx; k++) free((*images)[k]);
      free(*images); free(*labels);
      free(lbuf); fclose(fi); fclose(fl); return 0;
    }
    unsigned char lbl = lbuf[i];
#ifdef FULL_MNIST
    int keep = (lbl <= 9);
#else
    int keep = (lbl == 0 || lbl == 1);
#endif
    if (!keep) continue;

    (*images)[idx] = malloc(784 * sizeof(float));
    if (!(*images)[idx]) {
      for (int k = 0; k < idx; k++) free((*images)[k]);
      free(*images); free(*labels);
      free(lbuf); fclose(fi); fclose(fl); return 0;
    }
    (*labels)[idx] = (int)lbl;
    for (int j = 0; j < 784; j++)
      (*images)[idx][j] = (float)pbuf[j] / 255.0f;
    idx++;
  }

  free(lbuf);
  fclose(fi);
  fclose(fl);
  return 1;
}
