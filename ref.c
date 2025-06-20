#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 100
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 784  // 28x28 MNIST
#define LEARNING_RATE 0.01
#define EPOCHS 100
#define SAMPLES 1000

// Network weights and biases
float w1[INPUT_SIZE][HIDDEN_SIZE];
float b1[HIDDEN_SIZE];
float w2[HIDDEN_SIZE][OUTPUT_SIZE];
float b2[OUTPUT_SIZE];

// Activation functions
float sigmoid(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_deriv(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// Tanh activation (better for generators)
float tanh_activation(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    return tanhf(x);
}

float tanh_deriv(float x) {
    float t = tanh_activation(x);
    return 1.0f - t * t;
}

// Random number generation
float rand_float() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

// Xavier/Glorot initialization
void init_weights() {
    float w1_scale = sqrtf(2.0f / (INPUT_SIZE + HIDDEN_SIZE));
    float w2_scale = sqrtf(2.0f / (HIDDEN_SIZE + OUTPUT_SIZE));
    
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            w1[i][j] = rand_float() * w1_scale;
        }
    }
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        b1[j] = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            w2[j][k] = rand_float() * w2_scale;
        }
    }
    
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        b2[k] = 0.0f;
    }
}

// Forward pass
void forward(float *input, float *hidden, float *output) {
    // Input to hidden layer
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        hidden[j] = b1[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            hidden[j] += input[i] * w1[i][j];
        }
        hidden[j] = tanh_activation(hidden[j]);
    }

    // Hidden to output layer
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        output[k] = b2[k];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[k] += hidden[j] * w2[j][k];
        }
        output[k] = sigmoid(output[k]);  // Output in [0,1] range
    }
}

// Backpropagation
void backward(float *input, float *hidden, float *output, float *target, float *z_hidden) {
    float d_output[OUTPUT_SIZE];
    float d_hidden[HIDDEN_SIZE];

    // Calculate output layer gradients
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        d_output[k] = (output[k] - target[k]) * sigmoid_deriv(output[k]);
    }

    // Calculate hidden layer gradients
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float sum = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            sum += w2[j][k] * d_output[k];
        }
        d_hidden[j] = sum * tanh_deriv(z_hidden[j]);
    }

    // Update weights and biases for output layer
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            w2[j][k] -= LEARNING_RATE * hidden[j] * d_output[k];
        }
    }
    
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        b2[k] -= LEARNING_RATE * d_output[k];
    }

    // Update weights and biases for hidden layer
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            w1[i][j] -= LEARNING_RATE * input[i] * d_hidden[j];
        }
    }
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        b1[j] -= LEARNING_RATE * d_hidden[j];
    }
}

// Create synthetic training data (simple patterns)
void create_synthetic_data(float images[][784]) {
    for (int i = 0; i < SAMPLES; i++) {
        // Clear image
        for (int j = 0; j < 784; j++) {
            images[i][j] = 0.0f;
        }
        
        int pattern = i % 4;
        
        switch (pattern) {
            case 0: // Vertical line
                for (int row = 5; row < 23; row++) {
                    images[i][row * 28 + 14] = 1.0f;
                }
                break;
                
            case 1: // Horizontal line
                for (int col = 5; col < 23; col++) {
                    images[i][14 * 28 + col] = 1.0f;
                }
                break;
                
            case 2: // Circle
                for (int row = 0; row < 28; row++) {
                    for (int col = 0; col < 28; col++) {
                        float dx = col - 14;
                        float dy = row - 14;
                        float dist = sqrtf(dx*dx + dy*dy);
                        if (dist > 8 && dist < 10) {
                            images[i][row * 28 + col] = 1.0f;
                        }
                    }
                }
                break;
                
            case 3: // Square
                for (int row = 8; row < 20; row++) {
                    for (int col = 8; col < 20; col++) {
                        if (row == 8 || row == 19 || col == 8 || col == 19) {
                            images[i][row * 28 + col] = 1.0f;
                        }
                    }
                }
                break;
        }
        
        // Add some noise
        for (int j = 0; j < 784; j++) {
            if (rand_float() > 0.95f) {
                images[i][j] += 0.3f * rand_float();
                if (images[i][j] < 0) images[i][j] = 0;
                if (images[i][j] > 1) images[i][j] = 1;
            }
        }
    }
}

// Calculate mean squared error
float calculate_loss(float *output, float *target) {
    float loss = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / OUTPUT_SIZE;
}

// Training function
void train(float training_images[][784]) {
    float noise[INPUT_SIZE];
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
    float z_hidden[HIDDEN_SIZE];  // Pre-activation values for hidden layer
    
    printf("Starting training...\n");
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        
        for (int i = 0; i < SAMPLES; i++) {
            // Generate random noise input
            for (int j = 0; j < INPUT_SIZE; j++) {
                noise[j] = rand_float();
            }

            // Forward pass with z_hidden tracking
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                z_hidden[j] = b1[j];
                for (int k = 0; k < INPUT_SIZE; k++) {
                    z_hidden[j] += noise[k] * w1[k][j];
                }
                hidden[j] = tanh_activation(z_hidden[j]);
            }

            for (int k = 0; k < OUTPUT_SIZE; k++) {
                output[k] = b2[k];
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    output[k] += hidden[j] * w2[j][k];
                }
                output[k] = sigmoid(output[k]);
            }

            // Calculate loss
            total_loss += calculate_loss(output, training_images[i]);

            // Backward pass
            backward(noise, hidden, output, training_images[i], z_hidden);
        }
        
        if (epoch % 10 == 0) {
            printf("Epoch %d, Average Loss: %.6f\n", epoch + 1, total_loss / SAMPLES);
        }
    }
    
    printf("Training completed!\n");
}

// Generate and save image
void generate_image(const char *filename, int image_num) {
    float noise[INPUT_SIZE];
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
    float z_hidden[HIDDEN_SIZE];

    // Generate random noise
    for (int j = 0; j < INPUT_SIZE; j++) {
        noise[j] = rand_float();
    }

    // Forward pass
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        z_hidden[j] = b1[j];
        for (int k = 0; k < INPUT_SIZE; k++) {
            z_hidden[j] += noise[k] * w1[k][j];
        }
        hidden[j] = tanh_activation(z_hidden[j]);
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        output[k] = b2[k];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[k] += hidden[j] * w2[j][k];
        }
        output[k] = sigmoid(output[k]);
    }

    // Save as PGM format
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Error: Cannot create output file %s\n", filename);
        return;
    }
    
    fprintf(f, "P2\n28 28\n255\n");
    for (int i = 0; i < 784; i++) {
        int pixel = (int)(output[i] * 255.0f);
        if (pixel < 0) pixel = 0;
        if (pixel > 255) pixel = 255;
        fprintf(f, "%d ", pixel);
        if ((i + 1) % 28 == 0) {
            fprintf(f, "\n");
        }
    }
    fclose(f);
    
    printf("Generated image saved as %s\n", filename);
}

// Save training sample for comparison
void save_training_sample(float training_images[][784], int index, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Error: Cannot create sample file %s\n", filename);
        return;
    }
    
    fprintf(f, "P2\n28 28\n255\n");
    for (int i = 0; i < 784; i++) {
        int pixel = (int)(training_images[index][i] * 255.0f);
        if (pixel < 0) pixel = 0;
        if (pixel > 255) pixel = 255;
        fprintf(f, "%d ", pixel);
        if ((i + 1) % 28 == 0) {
            fprintf(f, "\n");
        }
    }
    fclose(f);
    
    printf("Training sample %d saved as %s\n", index, filename);
}

int main() {
    printf("MNIST-like Image Generator\n");
    printf("==========================\n");
    
    srand(42);
    
    // Initialize network
    printf("Initializing neural network...\n");
    init_weights();
    
    // Create synthetic training data
    printf("Creating synthetic training data...\n");
    float training_images[SAMPLES][784];
    create_synthetic_data(training_images);
    
    // Save some training samples for reference
    save_training_sample(training_images, 0, "training_sample_0.pgm");
    save_training_sample(training_images, 1, "training_sample_1.pgm");
    save_training_sample(training_images, 2, "training_sample_2.pgm");
    save_training_sample(training_images, 3, "training_sample_3.pgm");
    
    // Train the network
    train(training_images);
    
    // Generate multiple images
    printf("\nGenerating images...\n");
    for (int i = 0; i < 5; i++) {
        char filename[50];
        sprintf(filename, "generated_%d.pgm", i);
        generate_image(filename, i);
    }
    
    printf("\nAll done! Check the generated .pgm files.\n");
    printf("You can view them with image viewers that support PGM format,\n");
    printf("or convert them to PNG using: convert image.pgm image.png\n");
    
    return 0;
}