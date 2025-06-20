# MNIST Diffusion Model in C

A complete implementation of a diffusion model in pure C for generating handwritten digits (0 and 1) from the MNIST dataset.

## 🚀 Project Overview

This project implements a **diffusion model from scratch** using only standard C libraries (`stdlib.h` and `math.h`). The model learns to generate new images of handwritten digits 0 and 1 by training on filtered MNIST data.

### Key Features

- **Pure C Implementation**: No external dependencies except standard libraries
- **Neural Network from Scratch**: Manual implementation of forward and backward propagation
- **Diffusion Process**: Complete forward noise addition and reverse denoising
- **MNIST Integration**: Loads and filters real MNIST data for digits 0 and 1
- **Image Generation**: Generates new digit images and saves them as PGM files

## 🧠 Technical Details

### Architecture

- **Neural Network**: 2-layer MLP (784 → 256 → 784)
- **Activation**: ReLU for hidden layers
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: Mean Squared Error (MSE)
- **Diffusion Steps**: 100 timesteps with linear beta schedule

### Diffusion Process

1. **Forward Process**: Gradually adds Gaussian noise to training images
2. **Training**: Network learns to predict the noise added at each timestep
3. **Reverse Process**: Starts from pure noise and iteratively denoises to generate images

## 📋 Requirements

- GCC compiler with C99 support
- Math library (`-lm` flag)
- MNIST dataset files (optional - will use synthetic data if not available)

## 🛠️ Installation & Usage

### Quick Start (Synthetic Data)

```bash
# Compile the program
make

# Run with synthetic data
make run
```

### Using Real MNIST Data

1. **Download MNIST files** from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/):
   - `train-images-idx3-ubyte.gz`
   - `train-labels-idx1-ubyte.gz`

2. **Extract the files**:
   ```bash
   gunzip train-images-idx3-ubyte.gz
   gunzip train-labels-idx1-ubyte.gz
   ```

3. **Update the code** to use real MNIST files (modify `load_mnist_binary()` function)

4. **Compile and run**:
   ```bash
   make clean
   make
   ./diffusion_model
   ```

### Manual Compilation

```bash
gcc -Wall -Wextra -O3 -std=c99 -o diffusion_model diffusion_model.c -lm
```

## 📊 Output

The program generates:
- Training progress with loss values
- Generated images saved as `generated_0.pgm`, `generated_1.pgm`, etc.
- PGM files can be viewed with image viewers or converted to PNG/JPEG

### Viewing Generated Images

```bash
# Convert PGM to PNG (requires ImageMagick)
convert generated_0.pgm generated_0.png

# Or view directly with image viewers
eog generated_0.pgm  # Eye of GNOME
feh generated_0.pgm  # feh image viewer
```

## 🔧 Configuration

Key parameters can be adjusted in `diffusion_model.c`:

```c
#define IMAGE_SIZE 784      // 28x28 flattened
#define HIDDEN_SIZE 256     // Hidden layer size
#define TIMESTEPS 100       // Diffusion timesteps
#define LEARNING_RATE 0.001f // Learning rate
#define EPOCHS 100          // Training epochs
```

## 📁 Project Structure

```
├── diffusion_model.c    # Main implementation
├── mnist_loader.c       # MNIST data loader
├── Makefile            # Build configuration
├── README.md           # This file
└── generated_*.pgm     # Generated images (after running)
```

## 🎯 How It Works

### 1. Data Preparation
- Loads MNIST images filtered for digits 0 and 1
- Normalizes pixel values to [0, 1] range
- Creates synthetic data if MNIST files are unavailable

### 2. Neural Network
- **Input Layer**: 784 neurons (28×28 flattened image)
- **Hidden Layer**: 256 neurons with ReLU activation
- **Output Layer**: 784 neurons (predicted noise)

### 3. Diffusion Training
- Samples random timestep `t` and image `x₀`
- Adds noise: `x_t = √(ᾱ_t) * x₀ + √(1-ᾱ_t) * ε`
- Network predicts noise `ε`
- Minimizes MSE loss between predicted and true noise

### 4. Image Generation
- Starts with pure Gaussian noise
- Iteratively denoises for T timesteps
- Each step: `x_{t-1} = (x_t - √(1-ᾱ_t) * ε_θ(x_t,t)) / √(ᾱ_t)`

## 🧪 Mathematical Foundation

The model implements the DDPM (Denoising Diffusion Probabilistic Models) algorithm:

- **Forward Process**: `q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)`
- **Reverse Process**: `p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²I)`
- **Training Objective**: `L = E[||ε - ε_θ(x_t,t)||²]`

## 🎨 Customization

### Adding More Digits
Modify `load_mnist_binary()` to include more digit classes:

```c
if (label >= 0 && label <= 9) {  // All digits instead of just 0 and 1
    binary_count++;
}
```

### Changing Network Architecture
Adjust network size in the constants section:

```c
#define HIDDEN_SIZE 512  // Larger hidden layer
// Add more layers in forward_pass() and backward_pass()
```

### Different Noise Schedules
Modify the beta schedule in `create_scheduler()`:

```c
// Cosine schedule instead of linear
float cosine_schedule = cos(((float)t / TIMESTEPS + 0.008) / 1.008 * M_PI / 2);
scheduler->betas[t] = 0.0001 + 0.019 * (1 - cosine_schedule * cosine_schedule);
```

## 🐛 Troubleshooting

### Common Issues

1. **Compilation Errors**:
   - Ensure GCC supports C99: `gcc --version`
   - Link math library: `-lm` flag

2. **Poor Generation Quality**:
   - Increase training epochs
   - Adjust learning rate
   - Use real MNIST data instead of synthetic

3. **Memory Issues**:
   - Reduce `HIDDEN_SIZE` or `TIMESTEPS`
   - Check for memory leaks with valgrind

### Debug Mode
Add debug prints to monitor training:

```c
printf("Timestep: %d, Loss: %.6f\n", t, loss);
```

## 📈 Performance Tips

- **Optimization**: Use `-O3` flag for better performance
- **Memory**: Pre-allocate arrays to avoid frequent malloc/free
- **Vectorization**: Consider SIMD operations for matrix multiplication

## 🎓 Educational Value

This implementation demonstrates:
- **Deep Learning Fundamentals**: Backpropagation, gradient descent
- **Diffusion Models**: State-of-the-art generative modeling
- **Low-level Programming**: Memory management, numerical computation
- **Mathematical Implementation**: Translating theory to code

## 📚 References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Diffusion Models Tutorial](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

## 🤝 Contributing

Feel free to contribute improvements:
- Better optimization algorithms (Adam, RMSprop)
- More sophisticated network architectures
- Alternative noise schedules
- GPU acceleration with CUDA

## 📄 License

This project is released under the MIT License - see the code for details.

---

**Note**: This is a educational implementation. For production use, consider established frameworks like PyTorch or TensorFlow. 