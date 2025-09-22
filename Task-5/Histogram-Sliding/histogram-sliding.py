import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread(r'Task-5\Histogram-Sliding\dark-image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Sliding function
def histogram_slide(image, shift_value):
    slid = np.clip(image + shift_value, 0, 255)
    return slid.astype(np.uint8)

# Apply sliding (positive for brighter, negative for darker)
slid_image = histogram_slide(image, shift_value=17)

# Display results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title('Brightness Adjusted Image')
plt.imshow(slid_image)
plt.tight_layout()
plt.show()
