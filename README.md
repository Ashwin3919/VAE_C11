# MNIST VAE in C

A high-performance Variational Autoencoder (VAE) implementation in pure C for generating MNIST digits.

## 🚀 Features

- Clean, optimized C implementation
- No external dependencies (only standard C library + math)
- Fast training with optimized matrix operations
- High-quality digit generation
- Built-in MNIST data loader

## 📦 Project Structure

```
├── vae_model.c      # Main VAE implementation
├── mnist_loader.c   # MNIST data loader
├── Makefile        # Build configuration
├── README.md       # Documentation
└── data/           # MNIST dataset files
```

## 🛠️ Building

1. **Download MNIST data**:
   ```bash
   ./download_mnist.sh
   ```

2. **Build the VAE**:
   ```bash
   make clean
   make
   ```

3. **Run the model**:
   ```bash
   ./vae_model
   ```

## 🎯 Architecture

- **Encoder**: 784 → 512 → 32 (mean & logvar)
- **Latent**: 32-dimensional
- **Decoder**: 32 → 512 → 784
- ReLU activations + sigmoid output
- Batch normalization
- Optimized matrix operations

## 🔧 Configuration

Key parameters can be adjusted in `vae_model.c`:

```c
#define ENCODER_HIDDEN 512  // Encoder hidden layer size
#define LATENT_SIZE 32     // Latent space dimensionality
#define DECODER_HIDDEN 512  // Decoder hidden layer size
#define BATCH_SIZE 64      // Training batch size
#define LEARNING_RATE 0.001 // Learning rate
#define BETA 1.0           // KL divergence weight
```

## 📊 Output

The model generates:
- Training progress with reconstruction loss and KL divergence
- Generated digit samples saved as PGM files
- Real-time quality metrics

## 📝 License

MIT License - feel free to use and modify! 