#!/bin/bash

echo "🖼️  Converting PGM files to PNG format..."

# Create PNG directory if it doesn't exist
mkdir -p results_main/v3/png

# Counter for progress
total_files=$(ls results_main/v3/*.pgm 2>/dev/null | wc -l)
current=0

if [ "$total_files" -eq 0 ]; then
    echo "❌ No PGM files found in results/ directory"
    exit 1
fi

echo "Found $total_files PGM files to convert"
echo ""

# Convert all PGM files to PNG
for pgm_file in results_main/v3/*.pgm; do
    if [ -f "$pgm_file" ]; then
        # Get filename without path and extension
        filename=$(basename "$pgm_file" .pgm)
        png_file="results_main/v3/png/${filename}.png"
        
        # Convert using modern ImageMagick command
        magick "$pgm_file" "$png_file"
        
        current=$((current + 1))
        echo "✅ Converted: $filename.pgm → $filename.png ($current/$total_files)"
    fi
done

echo ""
echo "🎉 Conversion complete! PNG files are in results/png/"
echo ""
echo "📊 Quality progression samples created:"
echo "   • Epoch 0:   results/png/epoch_000_sample_0.png (initial noise)"
echo "   • Epoch 50:  results/png/epoch_050_sample_0.png (learning shapes)"
echo "   • Epoch 200: results/png/epoch_200_sample_0.png (high quality)"
echo ""
echo "💡 Open PNG files with:"
echo "   • Mac: open results/png/epoch_200_sample_0.png"
echo "   • Linux: xdg-open results/png/epoch_200_sample_0.png"
echo "   • Windows: start results/png/epoch_200_sample_0.png" 