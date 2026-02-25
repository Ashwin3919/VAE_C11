/*
 * view_results.c — ASCII-art PGM viewer for VAE-generated samples
 *
 * Usage: exe/view_results epoch_050_digit3_s0.pgm [more files...]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char SHADES[] = " .,:;oO#@";
#define N_SHADES ((int)(sizeof(SHADES) - 1))

static void view_pgm(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "cannot open: %s\n", filename);
    return;
  }

  char fmt[3] = {0};
  int  width = 0, height = 0, maxval = 0;

  /* width-limited %s prevents buffer overflow */
  if (fscanf(f, "%2s %d %d %d", fmt, &width, &height, &maxval) != 4 ||
      strcmp(fmt, "P2") != 0 || width != 28 || height != 28 || maxval < 1) {
    fprintf(stderr, "invalid PGM header: %s\n", filename);
    fclose(f);
    return;
  }

  printf("%s\n", filename);
  printf("+----------------------------+\n");
  for (int y = 0; y < height; y++) {
    printf("|");
    for (int x = 0; x < width; x++) {
      int pixel = 0;
      if (fscanf(f, "%d", &pixel) != 1) pixel = 0;
      pixel = pixel < 0 ? 0 : (pixel > maxval ? maxval : pixel);
      int idx = (int)((float)pixel / maxval * (N_SHADES - 1) + 0.5f);
      putchar(SHADES[idx]);
    }
    printf("|\n");
  }
  printf("+----------------------------+\n\n");
  fclose(f);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("VAE Results Viewer\n");
    printf("Usage: %s <pgm_file> [pgm_file ...]\n", argv[0]);
    printf("   or: %s results_main/v1/epoch_050_*.pgm\n", argv[0]);
    return 1;
  }

  printf("=== VAE Generated Samples ===\n\n");
  for (int i = 1; i < argc; i++)
    view_pgm(argv[i]);

  return 0;
}
