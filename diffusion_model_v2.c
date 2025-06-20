#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// Configuration constants - optimized for better results
#define IMAGE_SIZE 784  // 28x28 flattened
#define HIDDEN_SIZE 400
#define OUTPUT_SIZE 784
#define TIMESTEPS 20      // Fewer steps for better control
#define LEARNING_RATE 0.0003f
#define BATCH_SIZE 16
#define EPOCHS 500

// Neural network structure
typedef struct {
    float *weights_input_hidden;   
    float *bias_hidden;           
    float *weights_hidden_output; 
    float *bias_output;          
} Network;

// Diffusion scheduler
typedef struct {
    float *betas;           
    float *alphas;          
    float *alpha_cumprod;   
    float *sqrt_alpha_cumprod;       
    float *sqrt_one_minus_alpha_cumprod; 
    float *posterior_variance; 
} DiffusionScheduler;

// Training data structure
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
void forward_pass(Network *net, float *input, float *hidden, float *output);
void backward_pass(Network *net, float *input, float *hidden, float *output, 
                  float *target, float *grad_hidden, float *grad_input);
float relu(float x);
float relu_derivative(float x);
float tanh_activation(float x);
void add_noise(float *x0, float *noise, float *xt, int t, DiffusionScheduler *scheduler);
void generate_gaussian_noise(float *noise, int size);
Dataset* load_mnist_binary();
void free_dataset(Dataset *dataset);
void train_model(Network *net, DiffusionScheduler *scheduler, Dataset *dataset);
void sample_image(Network *net, DiffusionScheduler *scheduler, float *output);
void save_image_pgm(float *image, const char *filename);

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

// Network creation with better initialization
Network* create_network() {
    Network *net = (Network*)malloc(sizeof(Network));
    
    net->weights_input_hidden = (float*)malloc(IMAGE_SIZE * HIDDEN_SIZE * sizeof(float));
    net->bias_hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    net->weights_hidden_output = (float*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    net->bias_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Better initialization
    float scale_input = sqrtf(1.0f / IMAGE_SIZE);
    float scale_hidden = sqrtf(1.0f / HIDDEN_SIZE);
    
    for (int i = 0; i < IMAGE_SIZE * HIDDEN_SIZE; i++) {
        net->weights_input_hidden[i] = randn() * scale_input;
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        net->bias_hidden[i] = 0.0f;
    }
    
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        net->weights_hidden_output[i] = randn() * scale_hidden;
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
        free(net->weights_hidden_output);
        free(net->bias_output);
        free(net);
    }
}

// Improved scheduler
DiffusionScheduler* create_scheduler() {
    DiffusionScheduler *scheduler = (DiffusionScheduler*)malloc(sizeof(DiffusionScheduler));
    
    scheduler->betas = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->alphas = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->sqrt_alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->sqrt_one_minus_alpha_cumprod = (float*)malloc(TIMESTEPS * sizeof(float));
    scheduler->posterior_variance = (float*)malloc(TIMESTEPS * sizeof(float));
    
    // Better beta schedule
    float beta_start = 0.0001f;
    float beta_end = 0.02f;
    
    for (int t = 0; t < TIMESTEPS; t++) {
        float ratio = (float)t / (TIMESTEPS - 1);
        scheduler->betas[t] = beta_start + (beta_end - beta_start) * ratio * ratio;
        scheduler->alphas[t] = 1.0f - scheduler->betas[t];
    }
    
    scheduler->alpha_cumprod[0] = scheduler->alphas[0];
    for (int t = 1; t < TIMESTEPS; t++) {
        scheduler->alpha_cumprod[t] = scheduler->alpha_cumprod[t-1] * scheduler->alphas[t];
    }
    
    for (int t = 0; t < TIMESTEPS; t++) {
        scheduler->sqrt_alpha_cumprod[t] = sqrtf(scheduler->alpha_cumprod[t]);
        scheduler->sqrt_one_minus_alpha_cumprod[t] = sqrtf(1.0f - scheduler->alpha_cumprod[t]);
        
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
        free(scheduler);
    }
}

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

float tanh_activation(float x) {
    return tanhf(x);
}

// Forward pass with tanh output
void forward_pass(Network *net, float *input, float *hidden, float *output) {
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        hidden[h] = net->bias_hidden[h];
        for (int i = 0; i < IMAGE_SIZE; i++) {
            hidden[h] += input[i] * net->weights_input_hidden[i * HIDDEN_SIZE + h];
        }
        hidden[h] = relu(hidden[h]);
    }
    
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        output[o] = net->bias_output[o];
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            output[o] += hidden[h] * net->weights_hidden_output[h * OUTPUT_SIZE + o];
        }
        output[o] = tanh_activation(output[o]);
    }
}

// Backward pass 
void backward_pass(Network *net, float *input, float *hidden, float *output, 
                  float *target, float *grad_hidden, float *grad_input) {
    (void)grad_input;
    
    float *grad_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        grad_output[o] = 2.0f * (output[o] - target[o]) / OUTPUT_SIZE;
        grad_output[o] *= (1.0f - output[o] * output[o]);
    }
    
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            net->weights_hidden_output[h * OUTPUT_SIZE + o] -= 
                LEARNING_RATE * grad_output[o] * hidden[h];
        }
    }
    
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        net->bias_output[o] -= LEARNING_RATE * grad_output[o];
    }
    
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        grad_hidden[h] = 0.0f;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            grad_hidden[h] += grad_output[o] * net->weights_hidden_output[h * OUTPUT_SIZE + o];
        }
        grad_hidden[h] *= relu_derivative(hidden[h]);
    }
    
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
            
            // Normalize to [-1, 1] for better training with tanh
            for (int j = 0; j < 784; j++) {
                (*images)[binary_idx][j] = ((float)temp_image[j] / 255.0f) * 2.0f - 1.0f;
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
    printf("Loading MNIST data...\n");
    
    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    
    if (load_mnist_data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte",
                       &dataset->images, &dataset->labels, &dataset->count)) {
        return dataset;
    }
    
    printf("MNIST not found, creating synthetic data...\n");
    
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
                dataset->images[i][j] = (dist > 6 && dist < 10) ? 1.0f : -1.0f;
            } else {
                int col = j % 28;
                dataset->images[i][j] = (col > 12 && col < 16) ? 1.0f : -1.0f;
            }
            
            dataset->images[i][j] += 0.1f * randn();
            dataset->images[i][j] = fmaxf(-1.0f, fminf(1.0f, dataset->images[i][j]));
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

// Improved training
void train_model(Network *net, DiffusionScheduler *scheduler, Dataset *dataset) {
    printf("Training improved model...\n");
    
    float *hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *grad_hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *grad_input = (float*)malloc(IMAGE_SIZE * sizeof(float));
    float *noise = (float*)malloc(IMAGE_SIZE * sizeof(float));
    float *xt = (float*)malloc(IMAGE_SIZE * sizeof(float));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int num_samples = 0;
        
        int *indices = (int*)malloc(dataset->count * sizeof(int));
        for (int i = 0; i < dataset->count; i++) indices[i] = i;
        
        for (int i = dataset->count - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        for (int batch = 0; batch < dataset->count; batch += BATCH_SIZE) {
            for (int b = 0; b < BATCH_SIZE && batch + b < dataset->count; b++) {
                int idx = indices[batch + b];
                
                int t = rand() % TIMESTEPS;
                
                generate_gaussian_noise(noise, IMAGE_SIZE);
                add_noise(dataset->images[idx], noise, xt, t, scheduler);
                
                forward_pass(net, xt, hidden, output);
                
                float loss = 0.0f;
                for (int j = 0; j < IMAGE_SIZE; j++) {
                    float diff = output[j] - noise[j];
                    loss += diff * diff;
                }
                loss /= IMAGE_SIZE;
                total_loss += loss;
                num_samples++;
                
                backward_pass(net, xt, hidden, output, noise, grad_hidden, grad_input);
            }
        }
        
        free(indices);
        
        if (epoch % 50 == 0 || epoch < 10) {
            printf("Epoch %d/%d, Loss: %.6f\n", epoch, EPOCHS, total_loss / num_samples);
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

// Much improved sampling
void sample_image(Network *net, DiffusionScheduler *scheduler, float *output) {
    float *hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *predicted_noise = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    float *xt = (float*)malloc(IMAGE_SIZE * sizeof(float));
    
    generate_gaussian_noise(xt, IMAGE_SIZE);
    
    for (int t = TIMESTEPS - 1; t >= 0; t--) {
        forward_pass(net, xt, hidden, predicted_noise);
        
        if (t > 0) {
            float alpha_t = scheduler->alphas[t];
            float sqrt_alpha_t = sqrtf(alpha_t);
            float beta_t = scheduler->betas[t];
            
            for (int i = 0; i < IMAGE_SIZE; i++) {
                float pred_x0 = (xt[i] - scheduler->sqrt_one_minus_alpha_cumprod[t] * predicted_noise[i]) / 
                               scheduler->sqrt_alpha_cumprod[t];
                
                pred_x0 = fmaxf(-2.0f, fminf(2.0f, pred_x0));
                
                float posterior_mean = (sqrt_alpha_t * (1.0f - scheduler->alpha_cumprod[t-1]) * xt[i] + 
                                      scheduler->sqrt_alpha_cumprod[t-1] * beta_t * pred_x0) / 
                                     (1.0f - scheduler->alpha_cumprod[t]);
                
                float noise_scale = sqrtf(scheduler->posterior_variance[t]);
                xt[i] = posterior_mean + noise_scale * randn() * 0.3f;
            }
        } else {
            for (int i = 0; i < IMAGE_SIZE; i++) {
                output[i] = (xt[i] - scheduler->sqrt_one_minus_alpha_cumprod[t] * predicted_noise[i]) / 
                           scheduler->sqrt_alpha_cumprod[t];
            }
        }
    }
    
    // Convert to [0, 1]
    for (int i = 0; i < IMAGE_SIZE; i++) {
        output[i] = (output[i] + 1.0f) / 2.0f;
        output[i] = fmaxf(0.0f, fminf(1.0f, output[i]));
    }
    
    free(hidden);
    free(predicted_noise);
    free(xt);
}

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
}

int main() {
    printf("=== Improved MNIST Diffusion Model ===\n");
    
    srand(time(NULL));
    
    Network *net = create_network();
    DiffusionScheduler *scheduler = create_scheduler();
    Dataset *dataset = load_mnist_binary();
    
    train_model(net, scheduler, dataset);
    
    printf("\nGenerating improved samples...\n");
    float *generated_image = (float*)malloc(IMAGE_SIZE * sizeof(float));
    
    for (int i = 0; i < 8; i++) {
        sample_image(net, scheduler, generated_image);
        
        char filename[50];
        sprintf(filename, "improved_%d.pgm", i);
        save_image_pgm(generated_image, filename);
        printf("Generated %s\n", filename);
    }
    
    free(generated_image);
    free_dataset(dataset);
    free_scheduler(scheduler);
    free_network(net);
    
    printf("\nImproved diffusion model completed!\n");
    printf("Check improved_*.pgm files for better results.\n");
    
    return 0;
} 