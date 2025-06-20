#include <stdio.h>
#include <stdlib.h>

void display_pgm_ascii(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open %s\n", filename);
        return;
    }
    
    char format[3];
    int width, height, max_val;
    
    // Read PGM header
    fscanf(file, "%s", format);
    fscanf(file, "%d %d", &width, &height);
    fscanf(file, "%d", &max_val);
    
    printf("\n=== %s ===\n", filename);
    printf("Format: %s, Size: %dx%d, Max: %d\n\n", format, width, height, max_val);
    
    // ASCII characters for different intensity levels
    const char* ascii_chars = " .:-=+*#%@";
    int num_chars = 10;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixel;
            fscanf(file, "%d", &pixel);
            
            // Map pixel value to ASCII character
            int ascii_index = (pixel * (num_chars - 1)) / max_val;
            if (ascii_index >= num_chars) ascii_index = num_chars - 1;
            if (ascii_index < 0) ascii_index = 0;
            
            printf("%c", ascii_chars[ascii_index]);
        }
        printf("\n");
    }
    
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        // Display specific file
        display_pgm_ascii(argv[1]);
    } else {
        // Display all generated images
        printf("=== Generated Diffusion Model Images ===\n");
        
        for (int i = 0; i < 5; i++) {
            char filename[50];
            sprintf(filename, "generated_%d.pgm", i);
            display_pgm_ascii(filename);
            printf("\n");
        }
    }
    
    return 0;
} 