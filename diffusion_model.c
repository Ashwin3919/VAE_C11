#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Configuration constants
#define IMAGE_SIZE 784  // 28x28 flattened
#define HIDDEN_SIZE 512
#define HIDDEN_SIZE_2 256
#define OUTPUT_SIZE 784
#define TIMESTEPS 50
#define LEARNING_RATE 0.0005f
#define BATCH_SIZE 32
#define EPOCHS 300

// Neural network structure (3-layer network)
typedef struct {
    float *weights_input_hidden;   // [IMAGE_SIZE * HIDDEN_SIZE]
    float *bias_hidden;           // [HIDDEN_SIZE]
    float *weights_hidden_hidden2; // [HIDDEN_SIZE * HIDDEN_SIZE_2]
    float *bias_hidden2;          // [HIDDEN_SIZE_2]
    float *weights_hidden2_output; // [HIDDEN_SIZE_2 * OUTPUT_SIZE]
    float *bias_output;          // [OUTPUT_SIZE]
} Network;

// Diffusion scheduler
typedef struct {
    float *betas;           // [TIMESTEPS]
    float *alphas;          // [TIMESTEPS]
    float *alpha_cumprod;   // [TIMESTEPS]
    float *sqrt_alpha_cumprod;       // [TIMESTEPS]
    float *sqrt_one_minus_alpha_cumprod; // [TIMESTEPS]
} DiffusionScheduler;

// Training data structure
typedef struct {
    float **images;  // Array of image pointers
    int *labels;     // Array of labels (0 or 1)
    int count;       // Number of images
} Dataset;

// Function prototypes
Network* create_network();
void free_network(Network *net);
DiffusionScheduler* create_scheduler();
void free_scheduler(DiffusionScheduler *scheduler);
void forward_pass(Network *net, float *input, float *hidden, float *hidden2, float *output);
void backward_pass(Network *net, float *input, float *hidden, float *hidden2, float *output, 
                  float *target, float *grad_hidden, float *grad_hidden2, float *grad_input);
float relu(float x);
float relu_derivative(float x);
void add_noise(float *x0, float *noise, float *xt, int t, DiffusionScheduler *scheduler);
void generate_gaussian_noise(float *noise, int size);
Dataset* load_mnist_binary();
void free_dataset(Dataset *dataset);
void train_model(Network *net, DiffusionScheduler *scheduler, Dataset *dataset);
void sample_image(Network *net, DiffusionScheduler *scheduler, float *output);
void save_image_pgm(float *image, const char *filename);
void normalize_image(float *image);

// Random number generation
float randn() {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    
    has_spare = 1;
    float u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    float s = u * u + v * v;
    
    while (s >= 1.0f || s == 0.0f) {
        u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        s = u * u + v * v;
    }
    
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    return u * s;
}

// Network creation and destruction
Network* create_network() {
    Network *net = (Network*)malloc(sizeof(Network));
    
    net->weights_input_hidden = (float*)malloc(IMAGE_SIZE * HIDDEN_SIZE * sizeof(float));
    net->bias_hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    net->weights_hidden_hidden2 = (float*)malloc(HIDDEN_SIZE * HIDDEN_SIZE_2 * sizeof(float));
    net->bias_hidden2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
    net->weights_hidden2_output = (float*)malloc(HIDDEN_SIZE_2 * OUTPUT_SIZE * sizeof(float));
    net->bias_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Xavier initialization
    float scale_input = sqrtf(2.0f / IMAGE_SIZE);
    float scale_hidden = sqrtf(2.0f / HIDDEN_SIZE);
    float scale_hidden2 = sqrtf(2.0f / HIDDEN_SIZE_2);
    
    for (int i = 0; i < IMAGE_SIZE * HIDDEN_SIZE; i++) {
        net->weights_input_hidden[i] = randn() * scale_input;
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        net->bias_hidden[i] = 0.0f;
    }
    
    for (int i = 0; i < HIDDEN_SIZE * HIDDEN_SIZE_2; i++) {
        net->weights_hidden_hidden2[i] = randn() * scale_hidden;
    }
    
    for (int i = 0; i < HIDDEN_SIZE_2; i++) {
        net->bias_hidden2[i] = 0.0f;
    }
    
    for (int i = 0; i < HIDDEN_SIZE_2 * OUTPUT_SIZE; i++) {
        net->weights_hidden2_output[i] = randn() * scale_hidden2;
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        net->bias_output[i] = 0.0f;
    }
    
    return net;
}

void free_network(Network *net) {
    if (net) {
        free(net->weights_input_hidden);
        free(net->bias_hidden);
        free(net->weights_hidden_hidden2);
        free(net->bias_hidden2);
        free(net->weights_hidden2_output);
        free(net->bias_output);
        free(net);
    }
}

// Diffusion scheduler creation
DiffusionScheduler* create_scheduler() {
    DiffusionScheduler *scheduler = (DiffusionScheduler*)malloc(sizeof(DiffusionScheduler));
    
    scheduler->betas = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->alphas = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->sqrt_alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->sqrt_one_minus_alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    
    // Linear beta schedule
    float beta_start = 0.0001f;
    float beta_end = 0.02f;
    
    for (int t = 0; t < TIMESTEPS; t++) {
        scheduler->betas[t] = beta_start + (beta_end - beta_start) * t / (TIMESTEPS - 1);
        scheduler->alphas[t] = 1.0f - scheduler->betas[t];
    }
    
    // Compute cumulative products
    scheduler->alpha_cumprod[0] = scheduler->alphas[0];
    for (int t = 1; t < TIMESTEPS; t++) {
        scheduler->alpha_cumprod[t] = scheduler->alpha_cumprod[t-1] * scheduler->alphas[t];
    }
    
    // Precompute square roots for efficiency
    for (int t = 0; t < TIMESTEPS; t++) {
        scheduler->sqrt_alpha_cumprod[t] = sqrtf(scheduler->alpha_cumprod[t]);
        scheduler->sqrt_one_minus_alpha_cumprod[t] = sqrtf(1.0f - scheduler->alpha_cumprod[t]);
    }
    
    return scheduler;
}

void free_scheduler(DiffusionScheduler *scheduler) {
    if (scheduler) {
        free(scheduler->betas);
        free(scheduler->alphas);
        free(scheduler->alpha_cumprod);
        free(scheduler->sqrt_alpha_cumprod);
        free(scheduler->sqrt_one_minus_alpha_cumprod);
        free(scheduler);
    }
}

// Activation functions
float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// Neural network forward pass (3-layer network)
void forward_pass(Network *net, float *input, float *hidden, float *hidden2, float *output) {
    // Input to first hidden layer
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        hidden[h] = net->bias_hidden[h];
        for (int i = 0; i < IMAGE_SIZE; i++) {
            hidden[h] += input[i] * net->weights_input_hidden[i * HIDDEN_SIZE + h];
        }
        hidden[h] = relu(hidden[h]);
    }
    
    // First hidden to second hidden layer
    for (int h2 = 0; h2 < HIDDEN_SIZE_2; h2++) {
        hidden2[h2] = net->bias_hidden2[h2];
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            hidden2[h2] += hidden[h] * net->weights_hidden_hidden2[h * HIDDEN_SIZE_2 + h2];
        }
        hidden2[h2] = relu(hidden2[h2]);
    }
    
    // Second hidden to output layer (no activation for output)
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        output[o] = net->bias_output[o];
        for (int h2 = 0; h2 < HIDDEN_SIZE_2; h2++) {
            output[o] += hidden2[h2] * net->weights_hidden2_output[h2 * OUTPUT_SIZE + o];
        }
    }
}

// Neural network backward pass
void backward_pass(Network *net, float *input, float *hidden, float *output, 
                  float *target, float *grad_hidden, float *grad_input) {
    (void)grad_input; // Suppress unused parameter warning
    
    // Compute output gradients (MSE loss derivative)
    float *grad_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        grad_output[o] = 2.0f * (output[o] - target[o]) / OUTPUT_SIZE;
    }
    
    // Update output layer weights and biases
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            net->weights_hidden_output[h * OUTPUT_SIZE + o] -= 
                LEARNING_RATE * grad_output[o] * hidden[h];
        }
    }
    
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        net->bias_output[o] -= LEARNING_RATE * grad_output[o];
    }
    
    // Compute hidden gradients
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        grad_hidden[h] = 0.0f;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            grad_hidden[h] += grad_output[o] * net->weights_hidden_output[h * OUTPUT_SIZE + o];
        }
        grad_hidden[h] *= relu_derivative(hidden[h]);
    }
    
    // Update input layer weights and biases
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            net->weights_input_hidden[i * HIDDEN_SIZE + h] -= 
                LEARNING_RATE * grad_hidden[h] * input[i];
        }
    }
    
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        net->bias_hidden[h] -= LEARNING_RATE * grad_hidden[h];
    }
    
    free(grad_output);
}

// Add noise to image according to diffusion schedule
void add_noise(float *x0, float *noise, float *xt, int t, DiffusionScheduler *scheduler) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        xt[i] = scheduler->sqrt_alpha_cumprod[t] * x0[i] + 
                scheduler->sqrt_one_minus_alpha_cumprod[t] * noise[i];
    }
}

// Generate Gaussian noise
void generate_gaussian_noise(float *noise, int size) {
    for (int i = 0; i < size; i++) {
        noise[i] = randn();
    }
}

// Include the MNIST loader function
#include <stdint.h>

// Function prototypes for MNIST loading
uint32_t reverse_int(uint32_t i);
int load_mnist_data(const char* image_file, const char* label_file, 
                   float*** images, int** labels, int* count);

// MNIST file format constants
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_LABEL_MAGIC 0x00000801

// Function to reverse byte order for little-endian systems
uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

// Load MNIST images and filter for digits 0 and 1
int load_mnist_data(const char* image_file, const char* label_file, 
                   float*** images, int** labels, int* count) {
    
    FILE *img_fp = fopen(image_file, "rb");
    FILE *lbl_fp = fopen(label_file, "rb");
    
    if (!img_fp || !lbl_fp) {
        printf("Warning: Could not open MNIST files\n");
        printf("Expected files: %s and %s\n", image_file, label_file);
        if (img_fp) fclose(img_fp);
        if (lbl_fp) fclose(lbl_fp);
        return 0;
    }
    
    // Read image file header
    uint32_t magic, num_images, rows, cols;
    fread(&magic, sizeof(uint32_t), 1, img_fp);
    fread(&num_images, sizeof(uint32_t), 1, img_fp);
    fread(&rows, sizeof(uint32_t), 1, img_fp);
    fread(&cols, sizeof(uint32_t), 1, img_fp);
    
    magic = reverse_int(magic);
    num_images = reverse_int(num_images);
    rows = reverse_int(rows);
    cols = reverse_int(cols);
    
    printf("MNIST Images: %d, Size: %dx%d\n", num_images, rows, cols);
    
    if (magic != MNIST_IMAGE_MAGIC || rows != 28 || cols != 28) {
        printf("Error: Invalid MNIST image file format\n");
        fclose(img_fp);
        fclose(lbl_fp);
        return 0;
    }
    
    // Read label file header
    uint32_t label_magic, num_labels;
    fread(&label_magic, sizeof(uint32_t), 1, lbl_fp);
    fread(&num_labels, sizeof(uint32_t), 1, lbl_fp);
    
    label_magic = reverse_int(label_magic);
    num_labels = reverse_int(num_labels);
    
    if (label_magic != MNIST_LABEL_MAGIC || num_labels != num_images) {
        printf("Error: Invalid MNIST label file format\n");
        fclose(img_fp);
        fclose(lbl_fp);
        return 0;
    }
    
    // First pass: count images with labels 0 or 1
    int binary_count = 0;
    for (uint32_t i = 0; i < num_images; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, lbl_fp);
        if (label == 0 || label == 1) {
            binary_count++;
        }
    }
    
    printf("Found %d images with labels 0 or 1\n", binary_count);
    
    // Allocate memory
    *images = (float**)malloc(binary_count * sizeof(float*));
    *labels = (int*)malloc(binary_count * sizeof(int));
    *count = binary_count;
    
    // Reset file pointers
    fseek(img_fp, 4 * sizeof(uint32_t), SEEK_SET);
    fseek(lbl_fp, 2 * sizeof(uint32_t), SEEK_SET);
    
    // Second pass: load binary images
    int binary_idx = 0;
    unsigned char* temp_image = (unsigned char*)malloc(784 * sizeof(unsigned char));
    
    for (uint32_t i = 0; i < num_images; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, lbl_fp);
        fread(temp_image, sizeof(unsigned char), 784, img_fp);
        
        if (label == 0 || label == 1) {
            (*images)[binary_idx] = (float*)malloc(784 * sizeof(float));
            (*labels)[binary_idx] = (int)label;
            
            // Convert to float and normalize to [0, 1]
            for (int j = 0; j < 784; j++) {
                (*images)[binary_idx][j] = (float)temp_image[j] / 255.0f;
            }
            
            binary_idx++;
        }
    }
    
    free(temp_image);
    fclose(img_fp);
    fclose(lbl_fp);
    
    printf("Successfully loaded %d binary MNIST images\n", binary_count);
    return 1;
}

// Real MNIST data loader
Dataset* load_mnist_binary() {
    printf("Loading real MNIST data...\n");
    
    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    
    // Try to load real MNIST data first
    if (load_mnist_data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte",
                       &dataset->images, &dataset->labels, &dataset->count)) {
        printf("Using real MNIST training data\n");
        return dataset;
    }
    
    // Fallback to synthetic data if MNIST files not available
    printf("MNIST files not found, generating synthetic data...\n");
    
    dataset->count = 1000;  // Reduced for demo
    dataset->images = (float**)malloc(dataset->count * sizeof(float*));
    dataset->labels = (int*)malloc(dataset->count * sizeof(int));
    
    for (int i = 0; i < dataset->count; i++) {
        dataset->images[i] = (float*)malloc(IMAGE_SIZE * sizeof(float));
        dataset->labels[i] = rand() % 2;  // 0 or 1
        
        // Generate synthetic digit-like patterns
        for (int j = 0; j < IMAGE_SIZE; j++) {
            if (dataset->labels[i] == 0) {
                // Simple circle pattern for 0
                int row = j / 28;
                int col = j % 28;
                int center_row = 14, center_col = 14;
                float dist = sqrtf((row - center_row) * (row - center_row) + 
                                 (col - center_col) * (col - center_col));
                dataset->images[i][j] = (dist > 6 && dist < 10) ? 1.0f : 0.0f;
            } else {
                // Simple vertical line for 1
                int col = j % 28;
                dataset->images[i][j] = (col > 12 && col < 16) ? 1.0f : 0.0f;
            }
            
            // Add some noise
            dataset->images[i][j] += 0.1f * randn();
            dataset->images[i][j] = fmaxf(0.0f, fminf(1.0f, dataset->images[i][j]));
        }
    }
    
    printf("Generated %d synthetic images\n", dataset->count);
    return dataset;
}

void free_dataset(Dataset *dataset) {
    if (dataset) {
        for (int i = 0; i < dataset->count; i++) {
            free(dataset->images[i]);
        }
        free(dataset->images);
        free(dataset->labels);
        free(dataset);
    }
}

// Training function
void train_model(Network *net, DiffusionScheduler *scheduler, Dataset *dataset) {
    printf("Starting training...\n");
    
    float *hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *grad_hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *grad_input = (float*)malloc(IMAGE_SIZE * sizeof(float));
    float *noise = (float*)malloc(IMAGE_SIZE * sizeof(float));
    float *xt = (float*)malloc(IMAGE_SIZE * sizeof(float));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        for (int i = 0; i < dataset->count; i++) {
            // Sample random timestep
            int t = rand() % TIMESTEPS;
            
            // Generate noise and create noisy image
            generate_gaussian_noise(noise, IMAGE_SIZE);
            add_noise(dataset->images[i], noise, xt, t, scheduler);
            
            // Forward pass: predict the noise
            forward_pass(net, xt, hidden, output);
            
            // Compute loss (MSE between predicted and true noise)
            float loss = 0.0f;
            for (int j = 0; j < IMAGE_SIZE; j++) {
                float diff = output[j] - noise[j];
                loss += diff * diff;
            }
            loss /= IMAGE_SIZE;
            total_loss += loss;
            
            // Backward pass
            backward_pass(net, xt, hidden, output, noise, grad_hidden, grad_input);
            
            num_batches++;
        }
        
        if (epoch % 20 == 0 || epoch < 10) {
            printf("Epoch %d/%d, Average Loss: %.6f, Samples: %d\n", 
                   epoch, EPOCHS, total_loss / num_batches, num_batches);
        }
    }
    
    free(hidden);
    free(output);
    free(grad_hidden);
    free(grad_input);
    free(noise);
    free(xt);
    
    printf("Training completed!\n");
}

// Sampling function to generate new images
void sample_image(Network *net, DiffusionScheduler *scheduler, float *output) {
    float *hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *predicted_noise = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *xt = (float*)malloc(IMAGE_SIZE * sizeof(float));
    
    // Start with pure noise
    generate_gaussian_noise(xt, IMAGE_SIZE);
    
    // Reverse diffusion process
    for (int t = TIMESTEPS - 1; t >= 0; t--) {
        // Predict noise at time t
        forward_pass(net, xt, hidden, predicted_noise);
        
        // Denoise step: x_{t-1} = (x_t - sqrt(1-alpha_cumprod) * predicted_noise) / sqrt(alpha_cumprod)
        if (t > 0) {
            for (int i = 0; i < IMAGE_SIZE; i++) {
                xt[i] = (xt[i] - scheduler->sqrt_one_minus_alpha_cumprod[t] * predicted_noise[i]) / 
                        scheduler->sqrt_alpha_cumprod[t];
                
                // Add some noise for stochasticity (except at t=0)
                if (t > 0) {
                    xt[i] += sqrtf(scheduler->betas[t]) * randn();
                }
            }
        } else {
            // Final step, no additional noise
            for (int i = 0; i < IMAGE_SIZE; i++) {
                output[i] = (xt[i] - scheduler->sqrt_one_minus_alpha_cumprod[t] * predicted_noise[i]) / 
                           scheduler->sqrt_alpha_cumprod[t];
            }
        }
    }
    
    // Normalize output to [0, 1]
    normalize_image(output);
    
    free(hidden);
    free(predicted_noise);
    free(xt);
}

// Normalize image values to [0, 1]
void normalize_image(float *image) {
    float min_val = image[0], max_val = image[0];
    
    for (int i = 1; i < IMAGE_SIZE; i++) {
        if (image[i] < min_val) min_val = image[i];
        if (image[i] > max_val) max_val = image[i];
    }
    
    float range = max_val - min_val;
    if (range > 0) {
        for (int i = 0; i < IMAGE_SIZE; i++) {
            image[i] = (image[i] - min_val) / range;
        }
    }
}

// Save image as PGM format
void save_image_pgm(float *image, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    fprintf(file, "P2\n28 28\n255\n");
    
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            int pixel_value = (int)(image[i * 28 + j] * 255);
            pixel_value = pixel_value < 0 ? 0 : (pixel_value > 255 ? 255 : pixel_value);
            fprintf(file, "%d ", pixel_value);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    printf("Image saved as %s\n", filename);
}

// Main function
int main() {
    printf("=== MNIST Diffusion Model in C ===\n");
    
    // Initialize random seed
    srand(time(NULL));
    
    // Create network and scheduler
    Network *net = create_network();
    DiffusionScheduler *scheduler = create_scheduler();
    
    // Load dataset
    Dataset *dataset = load_mnist_binary();
    
    // Train the model
    train_model(net, scheduler, dataset);
    
    // Generate some sample images
    printf("\nGenerating sample images...\n");
    float *generated_image = (float*)malloc(IMAGE_SIZE * sizeof(float));
    
    for (int i = 0; i < 5; i++) {
        sample_image(net, scheduler, generated_image);
        
        char filename[50];
        sprintf(filename, "generated_%d.pgm", i);
        save_image_pgm(generated_image, filename);
    }
    
    // Cleanup
    free(generated_image);
    free_dataset(dataset);
    free_scheduler(scheduler);
    free_network(net);
    
    printf("\nDiffusion model training and generation completed!\n");
    printf("Check generated_*.pgm files for results.\n");
    
    return 0;
} 