CC      = gcc
CFLAGS  = -Wall -Wextra -O3 -std=c11 -march=native \
          -ffast-math -funroll-loops -ftree-vectorize \
          -flto -fomit-frame-pointer
LIBS    = -lm

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
CORE_SRCS = \
    $(SRC)/config/vae_config.c       \
    $(SRC)/rng/vae_rng.c             \
    $(SRC)/optimizer/vae_optimizer.c \
    $(SRC)/core/vae_model.c          \
    $(SRC)/io/vae_io.c               \
    $(SRC)/generate/vae_generate.c   \
    $(SRC)/train/vae_train.c         \
    $(SRC)/mnist_loader.c

MAIN_SRC = $(SRC)/main.c

# ── v1: small model, binary digits (0-1) ──────────────────────────────
all: $(EXE_DIR)/vae_model $(EXE_DIR)/view_results

$(EXE_DIR)/vae_model: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(IFLAGS) -o $@ $(MAIN_SRC) $(CORE_SRCS) $(LIBS)

# ── v2: medium model, binary digits (0-1) ─────────────────────────────
v2 mid: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(IFLAGS) -DVERSION_V2 -o $(EXE_DIR)/vae_model \
	    $(MAIN_SRC) $(CORE_SRCS) $(LIBS)

# ── v3: large model, full MNIST (0-9) ─────────────────────────────────
full: $(MAIN_SRC) $(CORE_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(IFLAGS) -DVERSION_V3 -DFULL_MNIST \
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

# ── results viewer (standalone, no VAE dependency) ────────────────────
$(EXE_DIR)/view_results: $(SRC)/view_results.c
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) -o $@ $^

# ── unit + integration tests (no MNIST required) ──────────────────────
# $(HEADERS) is listed as a prerequisite so that any public header change
# triggers a full test recompile, just as it does for the main binaries.
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
    $(SRC)/io/vae_io.c               \
    $(SRC)/generate/vae_generate.c   \
    $(SRC)/mnist_loader.c

test: $(TEST_SRCS) $(HEADERS)
	@mkdir -p $(EXE_DIR)
	$(CC) $(CFLAGS) $(TIFLAGS) -o $(EXE_DIR)/run_tests $(TEST_SRCS) $(LIBS)
	./$(EXE_DIR)/run_tests

# ── clean ──────────────────────────────────────────────────────────────
clean:
	rm -rf $(EXE_DIR) *.o *.pgm

.PHONY: all v2 mid full debug asan test clean
