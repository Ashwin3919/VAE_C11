# Hybrid MPI + OpenMP VAE Makefile
# Supports 4 modes: sequential, OpenMP-only, MPI-only, hybrid

# Compilation options - can be overridden via command line
USE_MPI ?= 1
USE_OPENMP ?= 1

# Base compiler and flags
# On macOS, use gcc-12 for OpenMP support, fallback to clang for non-OpenMP
BASE_CC = gcc
OPENMP_CC = gcc-12
CFLAGS = -Wall -Wextra -O3 -std=c99 -march=native -ffast-math
LIBS = -lm

# Detect if we're on macOS and adjust accordingly
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # Check if gcc-12 is available for OpenMP
    GCC12_AVAILABLE := $(shell which gcc-12 2>/dev/null)
    ifneq ($(GCC12_AVAILABLE),)
        OPENMP_CC = gcc-12
    else
        # Check for gcc-11, gcc-10, etc.
        GCC11_AVAILABLE := $(shell which gcc-11 2>/dev/null)
        ifneq ($(GCC11_AVAILABLE),)
            OPENMP_CC = gcc-11
        else
            # Fallback: try to use clang with libomp (if installed via homebrew)
            OPENMP_CC = clang
            LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
            ifneq ($(LIBOMP_PREFIX),)
                OPENMP_FLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
                OPENMP_LIBS = -L$(LIBOMP_PREFIX)/lib -lomp
            else
                OPENMP_FLAGS = -Xpreprocessor -fopenmp
                OPENMP_LIBS = -lomp
            endif
        endif
    endif
endif

# Conditional compiler and flag selection
ifeq ($(USE_MPI), 1)
    CC = mpicc
    CFLAGS += -DUSE_MPI=1
    ifeq ($(USE_OPENMP), 1)
        # Hybrid MPI + OpenMP mode
        ifdef OPENMP_FLAGS
            CFLAGS += $(OPENMP_FLAGS) -DUSE_OPENMP=1
            ifdef OPENMP_LIBS
                LIBS += $(OPENMP_LIBS)
            endif
        else
            CFLAGS += -fopenmp -DUSE_OPENMP=1
            LIBS += -lgomp
        endif
        MODE = hybrid
    else
        # MPI only mode
        CFLAGS += -DUSE_OPENMP=0
        MODE = mpi-only
    endif
else
    CFLAGS += -DUSE_MPI=0
    ifeq ($(USE_OPENMP), 1)
        # OpenMP only mode
        CC = $(OPENMP_CC)
        ifdef OPENMP_FLAGS
            CFLAGS += $(OPENMP_FLAGS) -DUSE_OPENMP=1
            ifdef OPENMP_LIBS
                LIBS += $(OPENMP_LIBS)
            endif
        else
            CFLAGS += -fopenmp -DUSE_OPENMP=1
            LIBS += -lgomp
        endif
        MODE = openmp-only
    else
        # Sequential mode
        CC = $(BASE_CC)
        CFLAGS += -DUSE_OPENMP=0
        MODE = sequential
    endif
endif

# Targets
all: vae_model view_results
	@echo "✅ Built VAE in $(MODE) mode"

# Main VAE model - hybrid parallel
vae_model: vae_model.c mnist_loader.c
	@echo "🔨 Compiling VAE in $(MODE) mode..."
	$(CC) $(CFLAGS) -o $@ vae_model.c $(LIBS)

# Results viewer
view_results: view_results.c
	$(BASE_CC) -Wall -Wextra -O2 -o $@ view_results.c

# Specific build targets for different modes
sequential:
	@echo "🚀 Building sequential version..."
	$(MAKE) clean
	$(MAKE) all USE_MPI=0 USE_OPENMP=0

openmp:
	@echo "🧮 Building OpenMP version..."
	$(MAKE) clean
	$(MAKE) all USE_MPI=0 USE_OPENMP=1

mpi:
	@echo "🌐 Building MPI version..."
	$(MAKE) clean
	$(MAKE) all USE_MPI=1 USE_OPENMP=0

hybrid:
	@echo "⚡ Building hybrid MPI+OpenMP version..."
	$(MAKE) clean
	$(MAKE) all USE_MPI=1 USE_OPENMP=1

# Parallelization parameters - optimized for your 8-core system
OPENMP_THREADS = 8
MPI_PROCESSES = 8
HYBRID_MPI_PROCESSES = 2
HYBRID_OPENMP_THREADS = 4

# Performance testing targets with specific thread/process counts
test-sequential:
	$(MAKE) sequential && time ./vae_model

test-openmp:
	$(MAKE) openmp && OMP_NUM_THREADS=$(OPENMP_THREADS) time ./vae_model

test-mpi:
	$(MAKE) mpi && time mpirun -np $(MPI_PROCESSES) ./vae_model

test-hybrid:
	$(MAKE) hybrid && OMP_NUM_THREADS=$(HYBRID_OPENMP_THREADS) time mpirun -np $(HYBRID_MPI_PROCESSES) ./vae_model

# Direct run targets (assumes already built)
run-sequential:
	@echo "🚀 Running sequential VAE..."
	./vae_model

run-openmp:
	@echo "🧮 Running OpenMP VAE with $(OPENMP_THREADS) threads..."
	OMP_NUM_THREADS=$(OPENMP_THREADS) ./vae_model

run-mpi:
	@echo "🌐 Running MPI VAE with $(MPI_PROCESSES) processes..."
	mpirun -np $(MPI_PROCESSES) ./vae_model

run-hybrid:
	@echo "⚡ Running hybrid VAE with $(HYBRID_MPI_PROCESSES) MPI processes × $(HYBRID_OPENMP_THREADS) OpenMP threads..."
	OMP_NUM_THREADS=$(HYBRID_OPENMP_THREADS) mpirun -np $(HYBRID_MPI_PROCESSES) ./vae_model

# Configuration targets to adjust parallelism
config-show:
	@echo "Current parallelization settings:"
	@echo "  OpenMP threads: $(OPENMP_THREADS)"
	@echo "  MPI processes: $(MPI_PROCESSES)"
	@echo "  Hybrid: $(HYBRID_MPI_PROCESSES) MPI × $(HYBRID_OPENMP_THREADS) OpenMP = $$(( $(HYBRID_MPI_PROCESSES) * $(HYBRID_OPENMP_THREADS) )) total cores"
	@echo "  System cores: $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'unknown')"

# Show current configuration
info:
	@echo "Current configuration:"
	@echo "  USE_MPI=$(USE_MPI)"
	@echo "  USE_OPENMP=$(USE_OPENMP)"
	@echo "  Mode: $(MODE)"
	@echo "  Compiler: $(CC)"
	@echo "  Flags: $(CFLAGS)"
	@echo "  Libraries: $(LIBS)"

# Clean rule
clean:
	rm -f vae_model view_results *.o *.pgm
	rm -rf results/

# Help target
help:
	@echo "VAE Hybrid Parallelization Build System"
	@echo "======================================="
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build with current settings"
	@echo "  sequential - Build sequential version"
	@echo "  openmp     - Build OpenMP-only version"
	@echo "  mpi        - Build MPI-only version"
	@echo "  hybrid     - Build hybrid MPI+OpenMP version"
	@echo ""
	@echo "Testing (Build + Run):"
	@echo "  test-sequential - Build and run sequential"
	@echo "  test-openmp     - Build and run OpenMP ($(OPENMP_THREADS) threads)"
	@echo "  test-mpi        - Build and run MPI ($(MPI_PROCESSES) processes)"
	@echo "  test-hybrid     - Build and run hybrid ($(HYBRID_MPI_PROCESSES) MPI × $(HYBRID_OPENMP_THREADS) OpenMP)"
	@echo ""
	@echo "Running (assumes built):"
	@echo "  run-sequential  - Run sequential version"
	@echo "  run-openmp      - Run OpenMP version ($(OPENMP_THREADS) threads)"
	@echo "  run-mpi         - Run MPI version ($(MPI_PROCESSES) processes)"
	@echo "  run-hybrid      - Run hybrid version ($(HYBRID_MPI_PROCESSES) MPI × $(HYBRID_OPENMP_THREADS) OpenMP)"
	@echo ""
	@echo "Utilities:"
	@echo "  config-show     - Show parallelization settings"
	@echo "  info            - Show current build configuration"
	@echo "  clean           - Remove built files"
	@echo "  help            - Show this help"
	@echo ""
	@echo "Manual override:"
	@echo "  make USE_MPI=1 USE_OPENMP=1  # Hybrid mode"
	@echo "  make USE_MPI=0 USE_OPENMP=1  # OpenMP only"
	@echo "  make USE_MPI=1 USE_OPENMP=0  # MPI only"
	@echo "  make USE_MPI=0 USE_OPENMP=0  # Sequential"

.PHONY: all sequential openmp mpi hybrid test-sequential test-openmp test-mpi test-hybrid run-sequential run-openmp run-mpi run-hybrid config-show info clean help 