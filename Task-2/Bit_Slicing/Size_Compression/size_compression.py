import cv2
import numpy as np
import os

# Load the image
input_path = input("Enter input image path: ").strip()
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

# No image warning
if img is None:
    raise ValueError("Image not found or unable to load.")

# Get original dimensions and size
original_shape = img.shape
original_size = os.path.getsize(input_path)

# Remove the Least Significant Bit (LSB)
# Right shift by 1 to remove LSB, then left shift to restore scale
img_7bit = (img >> 1) << 1

# Save the compressed image
output_path = input("Enter output image path to store: ").strip()
cv2.imwrite(output_path, img_7bit)

# Get new image dimensions and size
compressed_shape = img_7bit.shape
compressed_size = os.path.getsize(output_path)

# Calculate compression ratio
compression_ratio_in_percent = float((original_size-compressed_size) / original_size) * 100
compression_ratio = original_size / compressed_size

# Display results
print("Results\n\nImage Dimensions")
print(f"Original: {original_shape}")
print(f"Compressed: {compressed_shape}")

print("\nImage file Size")
print(f"Original: {original_size} bytes")
print(f"Compressed: {compressed_size} bytes")

print(f"\nCompression Ration: {compression_ratio:.2f}\nCompressed Image Size: {compression_ratio_in_percent:.2f} % smaller")
