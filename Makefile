CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99
LDFLAGS = -lm
TARGET = diffusion_model
TARGET_V2 = diffusion_model_v2
TARGET_V3 = diffusion_model_v3
TARGET_V3_2 = diffusion_model_v3_2
VIEWER = view_images
DATA_VIEWER = view_training_data
SOURCES = diffusion_model.c
SOURCES_V2 = diffusion_model_v2.c
SOURCES_V3 = diffusion_model_v3.c
SOURCES_V3_2 = diffusion_model_v3_2.c
VIEWER_SOURCES = view_images.c
DATA_VIEWER_SOURCES = view_training_data.c

# OpenMP flags for parallel version
OPENMP_FLAGS = -fopenmp -O3 -march=native -ffast-math -funroll-loops

all: $(TARGET) $(TARGET_V2) $(TARGET_V3) $(TARGET_V3_2) $(VIEWER) $(DATA_VIEWER)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

$(TARGET_V2): $(SOURCES_V2)
	$(CC) $(CFLAGS) -o $(TARGET_V2) $(SOURCES_V2) $(LDFLAGS)

$(TARGET_V3): $(SOURCES_V3)
	$(CC) $(CFLAGS) -o $(TARGET_V3) $(SOURCES_V3) $(LDFLAGS)

# Parallel version with pthreads optimization (macOS compatible)
$(TARGET_V3_2): $(SOURCES_V3_2)
	$(CC) $(CFLAGS) -O3 -march=native -ffast-math -funroll-loops -pthread -o $(TARGET_V3_2) $(SOURCES_V3_2) $(LDFLAGS)

$(VIEWER): $(VIEWER_SOURCES)
	$(CC) $(CFLAGS) -o $(VIEWER) $(VIEWER_SOURCES)

$(DATA_VIEWER): $(DATA_VIEWER_SOURCES)
	$(CC) $(CFLAGS) -o $(DATA_VIEWER) $(DATA_VIEWER_SOURCES)

clean:
	rm -f $(TARGET) $(TARGET_V2) $(TARGET_V3) $(TARGET_V3_2) $(VIEWER) $(DATA_VIEWER) generated_*.pgm improved_*.pgm training_*.pgm weights_*.bin realistic_*.pgm parallel_*.pgm

run: $(TARGET)
	./$(TARGET)

run-v2: $(TARGET_V2)
	./$(TARGET_V2)

run-v3: $(TARGET_V3)
	./$(TARGET_V3)

# Fast parallel training
run-parallel: $(TARGET_V3_2)
	@echo "🚀 Starting FAST parallel training..."
	OMP_NUM_THREADS=$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) ./$(TARGET_V3_2)

view: $(VIEWER)
	./$(VIEWER)

view-data: $(DATA_VIEWER)
	./$(DATA_VIEWER)

# Comprehensive test of all versions
test: all
	./$(TARGET) && ./$(VIEWER) generated_*.pgm

test-v2: all
	./$(TARGET_V2) && ./$(VIEWER) improved_*.pgm

test-v3: all
	./$(TARGET_V3) && ./$(VIEWER) realistic_*.pgm

test-parallel: all
	./$(TARGET_V3_2) && ./$(VIEWER) parallel_*.pgm

# Performance comparison
benchmark: all
	@echo "=== Performance Benchmark ==="
	@echo "Testing original model..."
	@time ./$(TARGET) > /dev/null
	@echo "Testing parallel model..."
	@time ./$(TARGET_V3_2) > /dev/null

# Show system info for optimization
sysinfo:
	@echo "=== System Information ==="
	@echo "CPU cores: $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
	@echo "OpenMP support: $(shell echo '#include <omp.h>' | $(CC) -fopenmp -E - >/dev/null 2>&1 && echo 'YES' || echo 'NO')"
	@echo "Compiler: $(CC) $(shell $(CC) --version | head -1)"

help:
	@echo "MNIST Diffusion Model - Available Commands:"
	@echo ""
	@echo "🚀 FAST TRAINING:"
	@echo "  make run-parallel  - Run parallel model (FASTEST)"
	@echo "  make test-parallel - Train + view results (parallel)"
	@echo ""
	@echo "Standard Training:"
	@echo "  make run           - Run original model"
	@echo "  make run-v2        - Run enhanced model v2" 
	@echo "  make run-v3        - Run enhanced model v3"
	@echo ""
	@echo "Testing & Viewing:"
	@echo "  make view-data     - View real MNIST training data"
	@echo "  make view          - View generated images"
	@echo "  make benchmark     - Performance comparison"
	@echo ""
	@echo "System:"
	@echo "  make sysinfo       - Show system optimization info"
	@echo "  make clean         - Remove executables and images"

.PHONY: all clean run run-v2 run-v3 run-parallel view view-data test test-v2 test-v3 test-parallel benchmark sysinfo help 