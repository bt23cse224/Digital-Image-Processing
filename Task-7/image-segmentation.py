import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, feature

# Load grayscale image
gray = data.cell() 
gray = np.uint8(gray)

# Compute Local Binary Pattern (LBP)
lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Define a simple kernel (3x3 averaging filter)
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=np.float32) / 9

# Multiply/Filter the LBP image with the kernel
segmented = cv2.filter2D(lbp, -1, kernel)

# Threshold to create a simple segmentation mask
_, mask = cv2.threshold(segmented, 128, 255, cv2.THRESH_BINARY)

# Results
titles = ['Original', 'LBP Image', 'After Kernel Filtering', 'Segmented Mask']
images = [gray, lbp, segmented, mask]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
