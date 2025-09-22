import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load grayscale images
source = cv2.imread(r'Task-5\Histogram-Specification\source.png', cv2.IMREAD_GRAYSCALE)
reference = cv2.imread(r'Task-5\Histogram-Specification\specification.png', cv2.IMREAD_GRAYSCALE)


#Compute histograms
def compute_histogram(image):
    hist = np.zeros(256)
    for pixel in image.flatten():
        hist[pixel] += 1
    return hist

source_hist = compute_histogram(source)
reference_hist = compute_histogram(reference)

#Normalize histograms to get probability distributions
source_pdf = source_hist / source.size
reference_pdf = reference_hist / reference.size

#Compute cumulative distribution functions
source_cdf = np.cumsum(source_pdf)
reference_cdf = np.cumsum(reference_pdf)

#Create a mapping from source to reference
mapping = np.zeros(256, dtype=np.uint8)
for src_val in range(256):
    # Find the reference pixel value with the closest CDF
    diff = np.abs(reference_cdf - source_cdf[src_val])
    mapping[src_val] = np.argmin(diff)

#Apply the mapping to the source image
matched_image = np.zeros_like(source)
for i in range(source.shape[0]):
    for j in range(source.shape[1]):
        matched_image[i, j] = mapping[source[i, j]]

#Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Source Image')
plt.imshow(source, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Specification Image')
plt.imshow(reference, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Matched Image')
plt.imshow(matched_image, cmap='gray')
plt.tight_layout()
plt.show()
