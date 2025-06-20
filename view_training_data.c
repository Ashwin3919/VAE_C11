#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// MNIST constants
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_LABEL_MAGIC 0x00000801

// Function to reverse byte order
uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

// Display image as ASCII art
void display_image_ascii(unsigned char *image, int label) {
    printf("\n=== MNIST Digit: %d ===\n", label);
    
    // ASCII characters for different intensity levels
    const char* ascii_chars = " .-:=+*#%@";
    int num_chars = 10;
    
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            int pixel = image[i * 28 + j];
            
            // Map pixel value to ASCII character
            int ascii_index = (pixel * (num_chars - 1)) / 255;
            if (ascii_index >= num_chars) ascii_index = num_chars - 1;
            if (ascii_index < 0) ascii_index = 0;
            
            printf("%c", ascii_chars[ascii_index]);
        }
        printf("\n");
    }
    printf("\n");
}

// Save image as PGM
void save_training_image_pgm(unsigned char *image, int label, int index) {
    char filename[50];
    sprintf(filename, "training_%d_%d.pgm", label, index);
    
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not create %s\n", filename);
        return;
    }
    
    fprintf(file, "P2\n28 28\n255\n");
    
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            fprintf(file, "%d ", image[i * 28 + j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    printf("Saved: %s\n", filename);
}

// Load and display MNIST training samples
int main() {
    printf("=== MNIST Training Data Viewer ===\n");
    
    // Try to open MNIST files
    FILE *img_fp = fopen("data/train-images-idx3-ubyte", "rb");
    FILE *lbl_fp = fopen("data/train-labels-idx1-ubyte", "rb");
    
    if (!img_fp || !lbl_fp) {
        printf("Error: Could not open MNIST files\n");
        printf("Expected files:\n");
        printf("- data/train-images-idx3-ubyte\n");
        printf("- data/train-labels-idx1-ubyte\n");
        printf("\nRun ./download_mnist.sh first to download the data.\n");
        if (img_fp) fclose(img_fp);
        if (lbl_fp) fclose(lbl_fp);
        return 1;
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
    
    printf("MNIST Dataset Info:\n");
    printf("- Total images: %d\n", num_images);
    printf("- Image size: %dx%d\n", rows, cols);
    
    // Read label file header
    uint32_t label_magic, num_labels;
    fread(&label_magic, sizeof(uint32_t), 1, lbl_fp);
    fread(&num_labels, sizeof(uint32_t), 1, lbl_fp);
    
    label_magic = reverse_int(label_magic);
    num_labels = reverse_int(num_labels);
    
    if (magic != MNIST_IMAGE_MAGIC || label_magic != MNIST_LABEL_MAGIC) {
        printf("Error: Invalid MNIST file format\n");
        fclose(img_fp);
        fclose(lbl_fp);
        return 1;
    }
    
    // Allocate memory for one image
    unsigned char *image = (unsigned char*)malloc(784 * sizeof(unsigned char));
    
    printf("\nLooking for digits 0 and 1...\n");
    
    int found_0s = 0, found_1s = 0;
    int max_examples = 5; // Show 5 examples of each digit
    
    // Scan through the dataset
    for (uint32_t i = 0; i < num_images && (found_0s < max_examples || found_1s < max_examples); i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, lbl_fp);
        fread(image, sizeof(unsigned char), 784, img_fp);
        
        if (label == 0 && found_0s < max_examples) {
            printf("\n==================================================");
            printf("\nTraining Example #%d (Label: %d)\n", i, label);
            display_image_ascii(image, label);
            save_training_image_pgm(image, label, found_0s);
            found_0s++;
        } else if (label == 1 && found_1s < max_examples) {
            printf("\n==================================================");
            printf("\nTraining Example #%d (Label: %d)\n", i, label);
            display_image_ascii(image, label);
            save_training_image_pgm(image, label, found_1s);
            found_1s++;
        }
    }
    
    printf("\n==================================================");
    printf("\nSummary:\n");
    printf("- Found %d examples of digit '0'\n", found_0s);
    printf("- Found %d examples of digit '1'\n", found_1s);
    printf("- Saved training images as training_*.pgm files\n");
    
    // Count total 0s and 1s in the dataset
    fseek(lbl_fp, 2 * sizeof(uint32_t), SEEK_SET); // Reset to start of labels
    int total_0s = 0, total_1s = 0;
    
    for (uint32_t i = 0; i < num_images; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, lbl_fp);
        if (label == 0) total_0s++;
        else if (label == 1) total_1s++;
    }
    
    printf("\nDataset Statistics:\n");
    printf("- Total 0s in dataset: %d\n", total_0s);
    printf("- Total 1s in dataset: %d\n", total_1s);
    printf("- Total binary samples: %d\n", total_0s + total_1s);
    printf("- Percentage of dataset: %.1f%%\n", 
           100.0f * (total_0s + total_1s) / num_images);
    
    // Cleanup
    free(image);
    fclose(img_fp);
    fclose(lbl_fp);
    
    printf("\nYou can view the saved images with:\n");
    printf("- ./view_images training_0_0.pgm\n");
    printf("- ./view_images training_1_0.pgm\n");
    printf("- Or convert to PNG: convert training_0_0.pgm training_0_0.png\n");
    
    return 0;
} 