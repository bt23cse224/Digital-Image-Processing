import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import pywt  # make sure PyWavelets is installed

# 1. Load grayscale image
img = data.coins()             
gray = np.uint8(img)            # convert to 8-bit

# 2. Perform 2D Discrete Wavelet Transform (DWT)
# IA = approximation (LL), IH = horizontal edges (LH)
# IV = vertical edges (HL), ID = diagonal edges (HH)
IA, (IH, IV, ID) = pywt.dwt2(gray, 'haar')

# 3.Enhance edges by multiplying detail coefficients to make edges stronger
IH_mod = IH * 2   # horizontal edges
IV_mod = IV * 2   # vertical edges
ID_mod = ID * 2   # diagonal edges

# 4. Reconstruct image using inverse DWT (IDWT)
reconstructed = pywt.idwt2((IA, (IH_mod, IV_mod, ID_mod)), 'haar')

# 5. Normalize the reconstructed image for display
reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 6. Display original, sub-bands, and reconstructed image
titles = [
    'Original',
    'Reconstructed with Enhanced Edges',
    'IA - Smooth (LL)',
    'IH - Horizontal Edges',
    'IV - Vertical Edges',
    'ID - Diagonal Edges'
]

images = [gray, reconstructed, IA, IH, IV, ID]

plt.figure(figsize=(14, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
