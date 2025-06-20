#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void view_pgm_as_ascii(const char* filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("❌ Cannot open %s\n", filename);
        return;
    }
    
    char format[3];
    int width, height, maxval;
    fscanf(f, "%s", format);
    fscanf(f, "%d %d", &width, &height);
    fscanf(f, "%d", &maxval);
    
    if (strcmp(format, "P2") != 0 || width != 28 || height != 28) {
        printf("❌ Invalid PGM format in %s\n", filename);
        fclose(f);
        return;
    }
    
    printf("📋 %s:\n", filename);
    printf("┌────────────────────────────┐\n");
    
    for (int y = 0; y < height; y++) {
        printf("│");
        for (int x = 0; x < width; x++) {
            int pixel;
            fscanf(f, "%d", &pixel);
            
            // Convert to ASCII art
            char c;
            if (pixel < 32) c = ' ';
            else if (pixel < 64) c = '.';
            else if (pixel < 96) c = ':';
            else if (pixel < 128) c = ';';
            else if (pixel < 160) c = 'o';
            else if (pixel < 192) c = 'O';
            else if (pixel < 224) c = '#';
            else c = '@';
            
            printf("%c", c);
        }
        printf("│\n");
    }
    printf("└────────────────────────────┘\n\n");
    
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("🖼️  VAE Results Viewer\n");
        printf("Usage: %s <pgm_file1> [pgm_file2] ...\n", argv[0]);
        printf("Or: %s results/epoch_*.pgm\n", argv[0]);
        return 1;
    }
    
    printf("🎨 VAE Generated Samples Viewer\n");
    printf("================================\n\n");
    
    for (int i = 1; i < argc; i++) {
        view_pgm_as_ascii(argv[i]);
    }
    
    return 0;
} 