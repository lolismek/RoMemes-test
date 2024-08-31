#!/bin/bash

# Directory containing the images
INPUT_DIR="corpus/images"
OUTPUT_DIR="ocr_results/tessdata-best_filters-50"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through all image files in the input directory
for IMAGE in "$INPUT_DIR"/*; do
    # Get the base name of the file (without directory and extension)
    BASENAME=$(basename "$IMAGE" | sed 's/\.[^.]*$//')
    EXTENSION="${IMAGE##*.}"
        
    ####################################################################################################
    input_image="$INPUT_DIR/$BASENAME.$EXTENSION"
    output_image="aux4.$EXTENSION"

    # Check if the image is a PNG and needs alpha channel removal
    if [ "$EXTENSION" == "png" ]; then
        # Remove alpha channel and save as output_image
        magick "$input_image" -background white -alpha remove -alpha off "$output_image"
    else
        # Copy the image to output_image (for non-PNG images)
        cp "$input_image" "$output_image"
    fi

    # 1. convert to dpi = 300 (if lower)
    desired_dpi=300

    current_dpi=$(identify -format "%x x %y" "$input_image")
    current_dpi_x=$(identify -format "%x" "$input_image" | sed 's/x.*//')

    if (( $(echo "$current_dpi_x < $desired_dpi" | bc -l) )); then
        magick "$input_image" -density $desired_dpi -units PixelsPerInch "$output_image"
    fi

    # 2. apply bilateral-blur
    magick "$output_image" -bilateral-blur 5x2 "$output_image"

    # 3. apply 10px black border
    magick "$output_image" -bordercolor black -border 10x10 "$output_image"

    # 4. convert to grayscale
    magick "$output_image" -colorspace Gray "$output_image"

    # 5. apply threshold
    magick "$output_image" -threshold 50% "$output_image"

    # 6. deskewing
    magick "$output_image" -deskew 40x40 "$output_image"
    ####################################################################################################


    # Perform OCR and save the output to a text file
    tesseract "$output_image" "$OUTPUT_DIR/$BASENAME" -l ron --psm 12 --tessdata-dir /opt/homebrew/share/tessdata_best
done



