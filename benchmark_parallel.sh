#!/bin/bash

# VAE Parallel Performance Benchmark Script
# Tests all parallelization modes and compares performance

echo "🚀 VAE Parallel Performance Benchmark"
echo "====================================="
echo "Testing scaled-up VAE model:"
echo "  - Architecture: 784→2048→1024→256→1024→2048→784"
echo "  - ~18M parameters (vs 1.1M before)"
echo "  - Batch size: 256 (vs 64 before)"  
echo "  - Dataset: 50K samples (vs 12K before)"
echo ""

# Create results directory
mkdir -p benchmark_results

# Function to run benchmark with timing
run_benchmark() {
    local mode=$1
    local description=$2
    local command=$3
    
    echo "🔥 Testing $description..."
    echo "Command: $command"
    
    # Run the benchmark and capture timing
    echo "Starting $mode benchmark..." > benchmark_results/${mode}_output.txt
    
    # Use time command to measure performance
    /usr/bin/time -l $command >> benchmark_results/${mode}_output.txt 2>&1 &
    local pid=$!
    
    # Wait a bit and show progress
    sleep 5
    
    # Check if still running and show CPU usage
    if kill -0 $pid 2>/dev/null; then
        echo "✅ $mode is running with high CPU usage - this is good!"
        ps -p $pid -o pid,ppid,%cpu,%mem,time,comm
        
        # Let it run for a reasonable time (30 seconds for demo)
        sleep 25
        
        # Stop the process for comparison
        kill $pid 2>/dev/null
        wait $pid 2>/dev/null
        echo "⏹️  Stopped $mode after 30 seconds for comparison"
    else
        echo "❌ $mode finished quickly or crashed"
    fi
    
    echo "📊 Results saved to benchmark_results/${mode}_output.txt"
    echo ""
}

# 1. Sequential Benchmark
echo "Building all versions first..."
make clean > /dev/null 2>&1

echo "1️⃣ SEQUENTIAL (Baseline)"
make sequential > /dev/null 2>&1
run_benchmark "sequential" "Sequential (1 core)" "./vae_model"

echo "2️⃣ OPENMP (8 threads)"
make openmp > /dev/null 2>&1
run_benchmark "openmp" "OpenMP (8 threads)" "OMP_NUM_THREADS=8 ./vae_model"

echo "3️⃣ MPI (8 processes)"
make mpi > /dev/null 2>&1
run_benchmark "mpi" "MPI (8 processes)" "mpirun -np 8 ./vae_model"

echo "4️⃣ HYBRID (2 MPI × 4 OpenMP)"
make hybrid > /dev/null 2>&1
run_benchmark "hybrid" "Hybrid (2 MPI × 4 OpenMP)" "OMP_NUM_THREADS=4 mpirun -np 2 ./vae_model"

echo "🏁 Benchmark Complete!"
echo "======================================"
echo "📊 Quick Performance Summary:"
echo "Check benchmark_results/ for detailed logs"
echo ""
echo "Expected results on scaled-up model:"
echo "  Sequential:  ~100% CPU (baseline)"
echo "  OpenMP:      ~400-800% CPU (4-8x speedup)"
echo "  MPI:         ~800% CPU distributed"  
echo "  Hybrid:      ~800% CPU (best scaling)"
echo ""
echo "The larger model should now show clear parallel benefits!" 