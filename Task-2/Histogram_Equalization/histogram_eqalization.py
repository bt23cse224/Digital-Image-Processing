import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
host_path = input("Enter host image path: ").strip()
img = cv2.imread(host_path, cv2.IMREAD_GRAYSCALE)

# Create histogram
hist, bins = np.histogram(img.flatten(), 256, [0,256])

# Compute CDF (cumulative distributive function)
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Mask all zeros (to avoid division by zero)
cdf_m = np.ma.masked_equal(cdf, 0)

# Histogram equalization formula
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')

# Apply the equalization
img_equalized = cdf_final[img]

# Compute histograms for original and equalized images
hist_orig = cv2.calcHist([img], [0], None, [256], [0,256])
hist_eq = cv2.calcHist([img_equalized], [0], None, [256], [0,256])

# Display images and histograms
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Original image
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# Equalized image
axs[0, 1].imshow(img_equalized, cmap='gray')
axs[0, 1].set_title('Equalized Image')
axs[0, 1].axis('off')

# Histogram of original image
axs[1, 0].plot(hist_orig, color='black')
axs[1, 0].set_title('Original Histogram')
axs[1, 0].set_xlim([0, 256])

# Histogram of equalized image
axs[1, 1].plot(hist_eq, color='black')
axs[1, 1].set_title('Equalized Histogram')
axs[1, 1].set_xlim([0, 256])

plt.tight_layout()
plt.show()
