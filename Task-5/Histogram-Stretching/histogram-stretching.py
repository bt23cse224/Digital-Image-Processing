import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread(r'Task-5\Histogram-Stretching\source-image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Stretching function
def histogram_stretch(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = ((image - min_val) / (max_val - min_val)) * 255
    return stretched.astype(np.uint8)

# Apply stretching
stretched_image = histogram_stretch(image)

# Display results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Stretched Image')
plt.imshow(stretched_image, cmap='gray')
plt.tight_layout()
plt.show()
