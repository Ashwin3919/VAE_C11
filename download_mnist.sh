#!/bin/bash

echo "Downloading MNIST dataset from reliable mirror..."

# Create data directory
mkdir -p data

# Use a more reliable mirror
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

# Download training data
echo "Downloading training images..."
curl -L ${BASE_URL}/train-images-idx3-ubyte.gz -o data/train-images-idx3-ubyte.gz

echo "Downloading training labels..."
curl -L ${BASE_URL}/train-labels-idx1-ubyte.gz -o data/train-labels-idx1-ubyte.gz

# Download test data
echo "Downloading test images..."
curl -L ${BASE_URL}/t10k-images-idx3-ubyte.gz -o data/t10k-images-idx3-ubyte.gz

echo "Downloading test labels..."
curl -L ${BASE_URL}/t10k-labels-idx1-ubyte.gz -o data/t10k-labels-idx1-ubyte.gz

echo "Extracting files..."
cd data

# Check if files were downloaded successfully
for file in *.gz; do
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo "Extracting $file..."
        gunzip -f "$file"
    else
        echo "Failed to download $file"
    fi
done

cd ..

echo "MNIST dataset files:"
ls -la data/

echo ""
echo "To use real MNIST data, the load_mnist_binary() function needs to be updated."
echo "The files are now available in the data/ directory." 