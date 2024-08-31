#!/bin/bash

# Directory containing the images
INPUT_DIR="corpus/images"
OUTPUT_DIR="ocr_results/raw_tessdata"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through all image files in the input directory
for IMAGE in "$INPUT_DIR"/*; do
    # Get the base name of the file (without directory and extension)
    BASENAME=$(basename "$IMAGE" | cut -d. -f1)
    
    # Perform OCR and save the output to a text file
    tesseract "$IMAGE" "$OUTPUT_DIR/$BASENAME" -l ron --psm 12 --tessdata-dir /opt/homebrew/share/tessdata --dpi 300
done

