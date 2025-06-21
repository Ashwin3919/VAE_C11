CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99 -march=native -ffast-math -funroll-loops -ftree-vectorize -flto -fomit-frame-pointer
LIBS = -lm

# Main target
all: vae_model view_results

# Build high-quality VAE
vae_model: vae_model.c mnist_loader.c
	$(CC) $(CFLAGS) -o $@ vae_model.c $(LIBS)

# Build results viewer
view_results: view_results.c
	$(CC) $(CFLAGS) -o $@ view_results.c

# Clean rule
clean:
	rm -f vae_model view_results *.o *.pgm

.PHONY: all clean 