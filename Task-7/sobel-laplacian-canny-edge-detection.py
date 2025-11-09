import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Sample image from skimage library
image = data.coins()  # sample coins RGB image
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert to BGR for OpenCV compatibility
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert standard bgr to rgb
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # convert rgb to grayscale

# 1. Sobel Edge Detection
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # detect edges in x-direction
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # detect edges in y-direction
sobel_combined = cv2.magnitude(sobelx, sobely)       # combine both directions

# 2. Laplacian Edge Detection
laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # detect edges using Laplacian operator

# 3. Canny Edge Detection
canny = cv2.Canny(gray, 100, 200)  # detect edges using thresholds

titles = ['Original', 'Sobel (Combined)', 'Laplacian', 'Canny']
images = [image, sobel_combined, laplacian, canny]

plt.figure(figsize=(12, 5))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
