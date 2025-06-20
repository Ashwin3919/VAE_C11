#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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
        printf("Error: Could not open MNIST files\n");
        printf("Please download MNIST files:\n");
        printf("- train-images-idx3-ubyte (or t10k-images-idx3-ubyte)\n");
        printf("- train-labels-idx1-ubyte (or t10k-labels-idx1-ubyte)\n");
        printf("From: http://yann.lecun.com/exdb/mnist/\n");
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
    
    printf("Image magic: %08x, Images: %d, Rows: %d, Cols: %d\n", 
           magic, num_images, rows, cols);
    
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
    
    printf("Label magic: %08x, Labels: %d\n", label_magic, num_labels);
    
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