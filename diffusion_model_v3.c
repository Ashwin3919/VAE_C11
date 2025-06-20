#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// Enhanced configuration for better results
#define IMAGE_SIZE 784  // 28x28 flattened
#define HIDDEN_SIZE_1 512
#define HIDDEN_SIZE_2 256  
#define HIDDEN_SIZE_3 128
#define OUTPUT_SIZE 784
#define TIMESTEPS 50
#define LEARNING_RATE 0.0001f
#define BATCH_SIZE 8
#define EPOCHS 1000

// Enhanced 4-layer neural network
typedef struct {
    // Layer 1: Input -> Hidden1
    float *weights_1;   // [IMAGE_SIZE * HIDDEN_SIZE_1]
    float *bias_1;      // [HIDDEN_SIZE_1]
    
    // Layer 2: Hidden1 -> Hidden2
    float *weights_2;   // [HIDDEN_SIZE_1 * HIDDEN_SIZE_2]
    float *bias_2;      // [HIDDEN_SIZE_2]
    
    // Layer 3: Hidden2 -> Hidden3
    float *weights_3;   // [HIDDEN_SIZE_2 * HIDDEN_SIZE_3]
    float *bias_3;      // [HIDDEN_SIZE_3]
    
    // Layer 4: Hidden3 -> Output
    float *weights_4;   // [HIDDEN_SIZE_3 * OUTPUT_SIZE]
    float *bias_4;      // [OUTPUT_SIZE]
} Network;

// Enhanced diffusion scheduler
typedef struct {
    float *betas;
    float *alphas;
    float *alpha_cumprod;
    float *sqrt_alpha_cumprod;
    float *sqrt_one_minus_alpha_cumprod;
    float *posterior_variance;
    float *sqrt_recip_alpha_cumprod;
    float *sqrt_recipm1_alpha_cumprod;
} DiffusionScheduler;

typedef struct {
    float **images;
    int *labels;
    int count;
} Dataset;

// Function prototypes
Network* create_network();
void free_network(Network *net);
DiffusionScheduler* create_scheduler();
void free_scheduler(DiffusionScheduler *scheduler);
void forward_pass(Network *net, float *input, float *h1, float *h2, float *h3, float *output);
void backward_pass(Network *net, float *input, float *h1, float *h2, float *h3, float *output, float *target);
float swish(float x);
float swish_derivative(float x);
void add_noise(float *x0, float *noise, float *xt, int t, DiffusionScheduler *scheduler);
void generate_gaussian_noise(float *noise, int size);
Dataset* load_mnist_binary();
void free_dataset(Dataset *dataset);
void train_model(Network *net, DiffusionScheduler *scheduler, Dataset *dataset);
void sample_image(Network *net, DiffusionScheduler *scheduler, float *output);
void save_image_pgm(float *image, const char *filename);

// Random number generation (Box-Muller)
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

// Enhanced activation function (Swish/SiLU)
float swish(float x) {
    return x / (1.0f + expf(-x));
}

float swish_derivative(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

// Create enhanced 4-layer network
Network* create_network() {
    Network *net = (Network*)malloc(sizeof(Network));
    
    // Allocate memory for all layers
    net->weights_1 = (float*)malloc(IMAGE_SIZE * HIDDEN_SIZE_1 * sizeof(float));
    net->bias_1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
    
    net->weights_2 = (float*)malloc(HIDDEN_SIZE_1 * HIDDEN_SIZE_2 * sizeof(float));
    net->bias_2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
    
    net->weights_3 = (float*)malloc(HIDDEN_SIZE_2 * HIDDEN_SIZE_3 * sizeof(float));
    net->bias_3 = (float*)malloc(HIDDEN_SIZE_3 * sizeof(float));
    
    net->weights_4 = (float*)malloc(HIDDEN_SIZE_3 * OUTPUT_SIZE * sizeof(float));
    net->bias_4 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // He initialization for better training
    float scale_1 = sqrtf(2.0f / IMAGE_SIZE);
    float scale_2 = sqrtf(2.0f / HIDDEN_SIZE_1);
    float scale_3 = sqrtf(2.0f / HIDDEN_SIZE_2);
    float scale_4 = sqrtf(2.0f / HIDDEN_SIZE_3);
    
    // Initialize weights
    for (int i = 0; i < IMAGE_SIZE * HIDDEN_SIZE_1; i++) {
        net->weights_1[i] = randn() * scale_1;
    }
    for (int i = 0; i < HIDDEN_SIZE_1 * HIDDEN_SIZE_2; i++) {
        net->weights_2[i] = randn() * scale_2;
    }
    for (int i = 0; i < HIDDEN_SIZE_2 * HIDDEN_SIZE_3; i++) {
        net->weights_3[i] = randn() * scale_3;
    }
    for (int i = 0; i < HIDDEN_SIZE_3 * OUTPUT_SIZE; i++) {
        net->weights_4[i] = randn() * scale_4;
    }
    
    // Initialize biases to zero
    for (int i = 0; i < HIDDEN_SIZE_1; i++) net->bias_1[i] = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE_2; i++) net->bias_2[i] = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE_3; i++) net->bias_3[i] = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) net->bias_4[i] = 0.0f;
    
    return net;
}

void free_network(Network *net) {
    if (net) {
        free(net->weights_1);
        free(net->bias_1);
        free(net->weights_2);
        free(net->bias_2);
        free(net->weights_3);
        free(net->bias_3);
        free(net->weights_4);
        free(net->bias_4);
        free(net);
    }
}

// Enhanced scheduler with better noise schedule
DiffusionScheduler* create_scheduler() {
    DiffusionScheduler *scheduler = (DiffusionScheduler*)malloc(sizeof(DiffusionScheduler));
    
    scheduler->betas = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->alphas = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->sqrt_alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->sqrt_one_minus_alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->posterior_variance = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->sqrt_recip_alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->sqrt_recipm1_alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    
    // Cosine schedule for better training
    float beta_start = 0.0001f;
    float beta_end = 0.02f;
    
    for (int t = 0; t < TIMESTEPS; t++) {
        float alpha_cumprod_t = cosf(((float)t / TIMESTEPS + 0.008f) / 1.008f * M_PI / 2.0f);
        alpha_cumprod_t = alpha_cumprod_t * alpha_cumprod_t;
        
        if (t == 0) {
            scheduler->alpha_cumprod[t] = alpha_cumprod_t;
            scheduler->betas[t] = 1.0f - alpha_cumprod_t;
        } else {
            scheduler->alpha_cumprod[t] = alpha_cumprod_t;
            scheduler->betas[t] = 1.0f - alpha_cumprod_t / scheduler->alpha_cumprod[t-1];
            scheduler->betas[t] = fminf(scheduler->betas[t], 0.999f);
        }
        
        scheduler->alphas[t] = 1.0f - scheduler->betas[t];
    }
    
    // Recompute alpha_cumprod properly
    scheduler->alpha_cumprod[0] = scheduler->alphas[0];
    for (int t = 1; t < TIMESTEPS; t++) {
        scheduler->alpha_cumprod[t] = scheduler->alpha_cumprod[t-1] * scheduler->alphas[t];
    }
    
    // Precompute useful values
    for (int t = 0; t < TIMESTEPS; t++) {
        scheduler->sqrt_alpha_cumprod[t] = sqrtf(scheduler->alpha_cumprod[t]);
        scheduler->sqrt_one_minus_alpha_cumprod[t] = sqrtf(1.0f - scheduler->alpha_cumprod[t]);
        scheduler->sqrt_recip_alpha_cumprod[t] = sqrtf(1.0f / scheduler->alpha_cumprod[t]);
        scheduler->sqrt_recipm1_alpha_cumprod[t] = sqrtf(1.0f / scheduler->alpha_cumprod[t] - 1.0f);
        
        if (t > 0) {
            scheduler->posterior_variance[t] = scheduler->betas[t] * 
                (1.0f - scheduler->alpha_cumprod[t-1]) / (1.0f - scheduler->alpha_cumprod[t]);
        } else {
            scheduler->posterior_variance[t] = scheduler->betas[t];
        }
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
        free(scheduler->posterior_variance);
        free(scheduler->sqrt_recip_alpha_cumprod);
        free(scheduler->sqrt_recipm1_alpha_cumprod);
        free(scheduler);
    }
}

// Enhanced 4-layer forward pass
void forward_pass(Network *net, float *input, float *h1, float *h2, float *h3, float *output) {
    // Layer 1: Input -> Hidden1
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        h1[i] = net->bias_1[i];
        for (int j = 0; j < IMAGE_SIZE; j++) {
            h1[i] += input[j] * net->weights_1[j * HIDDEN_SIZE_1 + i];
        }
        h1[i] = swish(h1[i]);
    }
    
    // Layer 2: Hidden1 -> Hidden2
    for (int i = 0; i < HIDDEN_SIZE_2; i++) {
        h2[i] = net->bias_2[i];
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            h2[i] += h1[j] * net->weights_2[j * HIDDEN_SIZE_2 + i];
        }
        h2[i] = swish(h2[i]);
    }
    
    // Layer 3: Hidden2 -> Hidden3
    for (int i = 0; i < HIDDEN_SIZE_3; i++) {
        h3[i] = net->bias_3[i];
        for (int j = 0; j < HIDDEN_SIZE_2; j++) {
            h3[i] += h2[j] * net->weights_3[j * HIDDEN_SIZE_3 + i];
        }
        h3[i] = swish(h3[i]);
    }
    
    // Layer 4: Hidden3 -> Output (no activation)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->bias_4[i];
        for (int j = 0; j < HIDDEN_SIZE_3; j++) {
            output[i] += h3[j] * net->weights_4[j * OUTPUT_SIZE + i];
        }
    }
}

// Enhanced backward pass with proper gradients
void backward_pass(Network *net, float *input, float *h1, float *h2, float *h3, float *output, float *target) {
    // Allocate gradient arrays
    float *grad_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *grad_h3 = (float*)malloc(HIDDEN_SIZE_3 * sizeof(float));
    float *grad_h2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
    float *grad_h1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
    
    // Compute output gradients
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        grad_output[i] = 2.0f * (output[i] - target[i]) / OUTPUT_SIZE;
    }
    
    // Backprop Layer 4: Output <- Hidden3
    for (int i = 0; i < HIDDEN_SIZE_3; i++) {
        grad_h3[i] = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            grad_h3[i] += grad_output[j] * net->weights_4[i * OUTPUT_SIZE + j];
            // Update weights
            net->weights_4[i * OUTPUT_SIZE + j] -= LEARNING_RATE * grad_output[j] * h3[i];
        }
        grad_h3[i] *= swish_derivative(h3[i]);
    }
    
    // Update output biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        net->bias_4[i] -= LEARNING_RATE * grad_output[i];
    }
    
    // Backprop Layer 3: Hidden3 <- Hidden2
    for (int i = 0; i < HIDDEN_SIZE_2; i++) {
        grad_h2[i] = 0.0f;
        for (int j = 0; j < HIDDEN_SIZE_3; j++) {
            grad_h2[i] += grad_h3[j] * net->weights_3[i * HIDDEN_SIZE_3 + j];
            // Update weights
            net->weights_3[i * HIDDEN_SIZE_3 + j] -= LEARNING_RATE * grad_h3[j] * h2[i];
        }
        grad_h2[i] *= swish_derivative(h2[i]);
    }
    
    // Update hidden3 biases
    for (int i = 0; i < HIDDEN_SIZE_3; i++) {
        net->bias_3[i] -= LEARNING_RATE * grad_h3[i];
    }
    
    // Backprop Layer 2: Hidden2 <- Hidden1
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        grad_h1[i] = 0.0f;
        for (int j = 0; j < HIDDEN_SIZE_2; j++) {
            grad_h1[i] += grad_h2[j] * net->weights_2[i * HIDDEN_SIZE_2 + j];
            // Update weights
            net->weights_2[i * HIDDEN_SIZE_2 + j] -= LEARNING_RATE * grad_h2[j] * h1[i];
        }
        grad_h1[i] *= swish_derivative(h1[i]);
    }
    
    // Update hidden2 biases
    for (int i = 0; i < HIDDEN_SIZE_2; i++) {
        net->bias_2[i] -= LEARNING_RATE * grad_h2[i];
    }
    
    // Backprop Layer 1: Hidden1 <- Input
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            // Update weights
            net->weights_1[i * HIDDEN_SIZE_1 + j] -= LEARNING_RATE * grad_h1[j] * input[i];
        }
    }
    
    // Update hidden1 biases
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        net->bias_1[i] -= LEARNING_RATE * grad_h1[i];
    }
    
    // Cleanup
    free(grad_output);
    free(grad_h3);
    free(grad_h2);
    free(grad_h1);
}

void add_noise(float *x0, float *noise, float *xt, int t, DiffusionScheduler *scheduler) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        xt[i] = scheduler->sqrt_alpha_cumprod[t] * x0[i] + 
                scheduler->sqrt_one_minus_alpha_cumprod[t] * noise[i];
    }
}

void generate_gaussian_noise(float *noise, int size) {
    for (int i = 0; i < size; i++) {
        noise[i] = randn();
    }
}

// MNIST loader
uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_LABEL_MAGIC 0x00000801

int load_mnist_data(const char* image_file, const char* label_file, 
                   float*** images, int** labels, int* count) {
    
    FILE *img_fp = fopen(image_file, "rb");
    FILE *lbl_fp = fopen(label_file, "rb");
    
    if (!img_fp || !lbl_fp) {
        if (img_fp) fclose(img_fp);
        if (lbl_fp) fclose(lbl_fp);
        return 0;
    }
    
    uint32_t magic, num_images, rows, cols;
    fread(&magic, sizeof(uint32_t), 1, img_fp);
    fread(&num_images, sizeof(uint32_t), 1, img_fp);
    fread(&rows, sizeof(uint32_t), 1, img_fp);
    fread(&cols, sizeof(uint32_t), 1, img_fp);
    
    magic = reverse_int(magic);
    num_images = reverse_int(num_images);
    rows = reverse_int(rows);
    cols = reverse_int(cols);
    
    uint32_t label_magic, num_labels;
    fread(&label_magic, sizeof(uint32_t), 1, lbl_fp);
    fread(&num_labels, sizeof(uint32_t), 1, lbl_fp);
    
    label_magic = reverse_int(label_magic);
    num_labels = reverse_int(num_labels);
    
    if (magic != MNIST_IMAGE_MAGIC || label_magic != MNIST_LABEL_MAGIC) {
        fclose(img_fp);
        fclose(lbl_fp);
        return 0;
    }
    
    int binary_count = 0;
    for (uint32_t i = 0; i < num_images; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, lbl_fp);
        if (label == 0 || label == 1) {
            binary_count++;
        }
    }
    
    *images = (float**)malloc(binary_count * sizeof(float*));
    *labels = (int*)malloc(binary_count * sizeof(int));
    *count = binary_count;
    
    fseek(img_fp, 4 * sizeof(uint32_t), SEEK_SET);
    fseek(lbl_fp, 2 * sizeof(uint32_t), SEEK_SET);
    
    int binary_idx = 0;
    unsigned char* temp_image = (unsigned char*)malloc(784 * sizeof(unsigned char));
    
    for (uint32_t i = 0; i < num_images; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, lbl_fp);
        fread(temp_image, sizeof(unsigned char), 784, img_fp);
        
        if (label == 0 || label == 1) {
            (*images)[binary_idx] = (float*)malloc(784 * sizeof(float));
            (*labels)[binary_idx] = (int)label;
            
            // Normalize to [0, 1] for better training
            for (int j = 0; j < 784; j++) {
                (*images)[binary_idx][j] = (float)temp_image[j] / 255.0f;
            }
            
            binary_idx++;
        }
    }
    
    free(temp_image);
    fclose(img_fp);
    fclose(lbl_fp);
    
    printf("Loaded %d MNIST images (0s and 1s)\n", binary_count);
    return 1;
}

Dataset* load_mnist_binary() {
    printf("Loading MNIST data for enhanced training...\n");
    
    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    
    if (load_mnist_data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte",
                       &dataset->images, &dataset->labels, &dataset->count)) {
        return dataset;
    }
    
    printf("MNIST not found, using synthetic data...\n");
    
    dataset->count = 2000;
    dataset->images = (float**)malloc(dataset->count * sizeof(float*));
    dataset->labels = (int*)malloc(dataset->count * sizeof(int));
    
    for (int i = 0; i < dataset->count; i++) {
        dataset->images[i] = (float*)malloc(IMAGE_SIZE * sizeof(float));
        dataset->labels[i] = rand() % 2;
        
        for (int j = 0; j < IMAGE_SIZE; j++) {
            if (dataset->labels[i] == 0) {
                int row = j / 28;
                int col = j % 28;
                int center_row = 14, center_col = 14;
                float dist = sqrtf((row - center_row) * (row - center_row) + 
                                 (col - center_col) * (col - center_col));
                dataset->images[i][j] = (dist > 6 && dist < 10) ? 1.0f : 0.0f;
            } else {
                int col = j % 28;
                dataset->images[i][j] = (col > 12 && col < 16) ? 1.0f : 0.0f;
            }
            
            dataset->images[i][j] += 0.05f * randn();
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

// Enhanced training with better optimization
void train_model(Network *net, DiffusionScheduler *scheduler, Dataset *dataset) {
    printf("Training enhanced 4-layer model for %d epochs...\n", EPOCHS);
    printf("Network: 784 -> %d -> %d -> %d -> 784\n", HIDDEN_SIZE_1, HIDDEN_SIZE_2, HIDDEN_SIZE_3);
    
    float *h1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
    float *h2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
    float *h3 = (float*)malloc(HIDDEN_SIZE_3 * sizeof(float));
    float *output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *noise = (float*)malloc(IMAGE_SIZE * sizeof(float));
    float *xt = (float*)malloc(IMAGE_SIZE * sizeof(float));
    
    float best_loss = 1e6f;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int num_samples = 0;
        
        // Shuffle dataset
        for (int i = dataset->count - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            
            // Swap images
            float *temp_img = dataset->images[i];
            dataset->images[i] = dataset->images[j];
            dataset->images[j] = temp_img;
            
            // Swap labels
            int temp_lbl = dataset->labels[i];
            dataset->labels[i] = dataset->labels[j];
            dataset->labels[j] = temp_lbl;
        }
        
        for (int i = 0; i < dataset->count; i += BATCH_SIZE) {
            for (int b = 0; b < BATCH_SIZE && i + b < dataset->count; b++) {
                int idx = i + b;
                
                // Sample random timestep
                int t = rand() % TIMESTEPS;
                
                // Generate noise and create noisy image
                generate_gaussian_noise(noise, IMAGE_SIZE);
                add_noise(dataset->images[idx], noise, xt, t, scheduler);
                
                // Forward pass: predict the noise
                forward_pass(net, xt, h1, h2, h3, output);
                
                // Compute loss
                float loss = 0.0f;
                for (int j = 0; j < IMAGE_SIZE; j++) {
                    float diff = output[j] - noise[j];
                    loss += diff * diff;
                }
                loss /= IMAGE_SIZE;
                total_loss += loss;
                num_samples++;
                
                // Backward pass
                backward_pass(net, xt, h1, h2, h3, output, noise);
            }
        }
        
        float avg_loss = total_loss / num_samples;
        
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
        }
        
        if (epoch % 100 == 0 || epoch < 20) {
            printf("Epoch %d/%d, Loss: %.6f, Best: %.6f\n", 
                   epoch, EPOCHS, avg_loss, best_loss);
        }
        
        // Early stopping if loss gets very small
        if (avg_loss < 0.001f) {
            printf("Early stopping at epoch %d (loss: %.6f)\n", epoch, avg_loss);
            break;
        }
    }
    
    free(h1);
    free(h2);
    free(h3);
    free(output);
    free(noise);
    free(xt);
    
    printf("Training completed! Best loss: %.6f\n", best_loss);
}

// Enhanced sampling with DDPM algorithm
void sample_image(Network *net, DiffusionScheduler *scheduler, float *output) {
    float *h1 = (float*)malloc(HIDDEN_SIZE_1 * sizeof(float));
    float *h2 = (float*)malloc(HIDDEN_SIZE_2 * sizeof(float));
    float *h3 = (float*)malloc(HIDDEN_SIZE_3 * sizeof(float));
    float *predicted_noise = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *xt = (float*)malloc(IMAGE_SIZE * sizeof(float));
    
    // Start with pure noise
    generate_gaussian_noise(xt, IMAGE_SIZE);
    
    // DDPM reverse process
    for (int t = TIMESTEPS - 1; t >= 0; t--) {
        // Predict noise
        forward_pass(net, xt, h1, h2, h3, predicted_noise);
        
        if (t > 0) {
            // Use proper DDPM formula
            for (int i = 0; i < IMAGE_SIZE; i++) {
                // Predict x0
                float pred_x0 = scheduler->sqrt_recip_alpha_cumprod[t] * xt[i] - 
                               scheduler->sqrt_recipm1_alpha_cumprod[t] * predicted_noise[i];
                
                // Clamp x0 to reasonable range
                pred_x0 = fmaxf(0.0f, fminf(1.0f, pred_x0));
                
                // Compute posterior mean
                float posterior_mean = scheduler->sqrt_alpha_cumprod[t-1] * scheduler->betas[t] * pred_x0 / 
                                     (1.0f - scheduler->alpha_cumprod[t]) + 
                                     scheduler->sqrt_alpha_cumprod[t] * (1.0f - scheduler->alpha_cumprod[t-1]) * xt[i] / 
                                     (1.0f - scheduler->alpha_cumprod[t]);
                
                // Add noise
                float posterior_std = sqrtf(scheduler->posterior_variance[t]);
                xt[i] = posterior_mean + posterior_std * randn() * 0.5f;
            }
        } else {
            // Final step
            for (int i = 0; i < IMAGE_SIZE; i++) {
                output[i] = scheduler->sqrt_recip_alpha_cumprod[t] * xt[i] - 
                           scheduler->sqrt_recipm1_alpha_cumprod[t] * predicted_noise[i];
                output[i] = fmaxf(0.0f, fminf(1.0f, output[i]));
            }
        }
    }
    
    free(h1);
    free(h2);
    free(h3);
    free(predicted_noise);
    free(xt);
}

void save_image_pgm(float *image, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not create %s\n", filename);
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
}

int main() {
    printf("=== ENHANCED MNIST Diffusion Model v3 ===\n");
    printf("Target: Generate realistic digits 0 and 1\n\n");
    
    srand(time(NULL));
    
    Network *net = create_network();
    DiffusionScheduler *scheduler = create_scheduler();
    Dataset *dataset = load_mnist_binary();
    
    printf("Dataset loaded: %d samples\n", dataset->count);
    printf("Starting intensive training...\n\n");
    
    train_model(net, scheduler, dataset);
    
    printf("\nGenerating realistic digits...\n");
    float *generated_image = (float*)malloc(IMAGE_SIZE * sizeof(float));
    
    for (int i = 0; i < 10; i++) {
        sample_image(net, scheduler, generated_image);
        
        char filename[50];
        sprintf(filename, "realistic_%d.pgm", i);
        save_image_pgm(generated_image, filename);
        printf("Generated: %s\n", filename);
    }
    
    free(generated_image);
    free_dataset(dataset);
    free_scheduler(scheduler);
    free_network(net);
    
    printf("\n🎉 Enhanced diffusion model completed!\n");
    printf("Check realistic_*.pgm files for high-quality results.\n");
    printf("Compare with real MNIST using: ./view_images realistic_0.pgm\n");
    
    return 0;
} 