CC      = gcc
CFLAGS  = -Wall -Wextra -O3 -std=c11 -march=native \
          -ffast-math -funroll-loops -ftree-vectorize \
          -flto -fomit-frame-pointer
LIBS    = -lm

# ── OpenMP (Apple Clang requires explicit libomp from Homebrew) ────────
# Override on Linux: make omp-mini LIBOMP_CFLAGS=-fopenmp LIBOMP_LIBS=-lgomp
LIBOMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
LIBOMP_CFLAGS  = -Xclang -fopenmp -I$(LIBOMP_PREFIX)/include
LIBOMP_LIBS    = -L$(LIBOMP_PREFIX)/lib -lomp

# ── directories ────────────────────────────────────────────────────────
EXE_DIR = exe
INC     = include
SRC     = src
TESTS   = tests

# ── include flags ──────────────────────────────────────────────────────
IFLAGS  = -I$(INC)
TIFLAGS = -I$(INC) -I$(TESTS)

# ── public headers ─────────────────────────────────────────────────────
# Every target lists $(HEADERS) as a prerequisite.  Any header change
# forces a full rebuild of every binary that transitively includes it,
# preventing stale-object bugs without requiring per-TU dependency files.
HEADERS = $(wildcard $(INC)/*.h)

# ── core implementation files (shared by all binary targets and tests) ─
# vae_model.c  : slab lifecycle only (create_vae / free_vae)
# vae_forward.c: encoder + decoder forward pass
# vae_backward.c: gradient accumulation
# vae_loss.c   : ELBO loss + Adam update + vae_decode
CORE_SRCS = \
    $(SRC)/config/vae_config.c       \
    $(SRC)/rng/vae_rng.c             \
    $(SRC)/optimizer/vae_optimizer.c \
    $(SRC)/core/vae_model.c          \
    $(SRC)/core/vae_forward.c        \
    $(SRC)/core/vae_backward.c       \
    $(SRC)/core/vae_loss.c           \
    $(SRC)/io/vae_io.c               \
    $(SRC)/generate/vae_generate.c   \
    $(SRC)/train/vae_train.c         \
    $(SRC)/mnist_loader.c

MAIN_SRC = $(SRC)/main.c

# ── v1: small model, binary digits (0-1) ──────────────────────────────
mini: all
all: $(EXE_DIR)/vae_model $(EXE_DIR)/view_results

$(EXE_DIR)/vae_model: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(IFLAGS) -o $@ $(MAIN_SRC) $(CORE_SRCS) $(LIBS)

# ── v2: medium model, binary digits (0-1) ─────────────────────────────
v2: mid
mid: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(IFLAGS) -DVERSION_V2 -o $(EXE_DIR)/vae_model \
	    $(MAIN_SRC) $(CORE_SRCS) $(LIBS)

# ── v3: large model, full MNIST (0-9) ─────────────────────────────────
# Note: -DFULL_MNIST removed; digit mode is now a runtime flag (--full-mnist)
full: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(IFLAGS) -DVERSION_V3 \
	    -o $(EXE_DIR)/vae_model $(MAIN_SRC) $(CORE_SRCS) $(LIBS)

# ── debug build ────────────────────────────────────────────────────────
debug: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) -g -O0 -std=c11 -DDEBUG $(IFLAGS) \
	    -o $(EXE_DIR)/vae_model_debug $(MAIN_SRC) $(CORE_SRCS) $(LIBS)

# ── sanitizer build (AddressSanitizer + UBSan) ─────────────────────────
# Run with:  make asan && ./exe/vae_model_asan
# Catches: heap/stack overflows, use-after-free, signed overflow, null deref.
asan: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) -g -O1 -std=c11 -fsanitize=address,undefined \
	    -fno-omit-frame-pointer $(IFLAGS) \
	    -o $(EXE_DIR)/vae_model_asan $(MAIN_SRC) $(CORE_SRCS) $(LIBS)

# ── OpenMP builds — activates #pragma omp parallel for in linear_batch.
# Three variants so training throughput can be compared across model sizes.
# Usage: make omp        (builds all three)
#        make omp-mini / omp-mid / omp-full   (individual)
omp-mini: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(IFLAGS) $(LIBOMP_CFLAGS) \
	    -o $(EXE_DIR)/vae_model_omp_mini $(MAIN_SRC) $(CORE_SRCS) $(LIBS) $(LIBOMP_LIBS)

omp-mid: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(IFLAGS) $(LIBOMP_CFLAGS) -DVERSION_V2 \
	    -o $(EXE_DIR)/vae_model_omp_mid $(MAIN_SRC) $(CORE_SRCS) $(LIBS) $(LIBOMP_LIBS)

omp-full: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(IFLAGS) $(LIBOMP_CFLAGS) -DVERSION_V3 \
	    -o $(EXE_DIR)/vae_model_omp_full $(MAIN_SRC) $(CORE_SRCS) $(LIBS) $(LIBOMP_LIBS)

# Convenience alias — builds all three OMP variants at once.
omp: omp-mini omp-mid omp-full

# ── results viewer (standalone, no VAE dependency) ────────────────────
$(EXE_DIR)/view_results: $(SRC)/view_results.c
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) -o $@ $^

# ── test sources (shared between test and tsan targets) ───────────────
TEST_SRCS = \
    $(TESTS)/run_tests.c             \
    $(TESTS)/test_rng.c              \
    $(TESTS)/test_optimizer.c        \
    $(TESTS)/test_model.c            \
    $(TESTS)/test_train.c            \
    $(SRC)/config/vae_config.c       \
    $(SRC)/rng/vae_rng.c             \
    $(SRC)/optimizer/vae_optimizer.c \
    $(SRC)/core/vae_model.c          \
    $(SRC)/core/vae_forward.c        \
    $(SRC)/core/vae_backward.c       \
    $(SRC)/core/vae_loss.c           \
    $(SRC)/io/vae_io.c               \
    $(SRC)/generate/vae_generate.c   \
    $(SRC)/mnist_loader.c

# ── unit + integration tests (no MNIST required) ──────────────────────
# $(HEADERS) is listed as a prerequisite so that any public header change
# triggers a full test recompile, just as it does for the main binaries.
test: $(TEST_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(TIFLAGS) -o $(EXE_DIR)/run_tests $(TEST_SRCS) $(LIBS)
	./$(EXE_DIR)/run_tests

# ── ThreadSanitizer — validate OMP paths for data races ───────────────
# Run on Linux with gcc; on macOS clang TSan is also supported.
# Catches races in: linear_batch parallel for, rng_normal concurrent calls.
tsan: $(TEST_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) -g -O1 -std=c11 -fsanitize=thread \
	    -fno-omit-frame-pointer $(TIFLAGS) \
	    -o $(EXE_DIR)/run_tests_tsan $(TEST_SRCS) $(LIBS)
	./$(EXE_DIR)/run_tests_tsan

# ── clean ──────────────────────────────────────────────────────────────
clean:
	rm -rf $(EXE_DIR) *.o *.pgm results_main models

.PHONY: all mini v2 mid full debug asan omp omp-mini omp-mid omp-full test tsan clean
