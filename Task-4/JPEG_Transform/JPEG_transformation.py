import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('Task-4\JPEG_Transform\original.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert standard bgr to rgb

# Encode image to JPEG format with specified quality
encoding_parameter = [int(cv2.IMWRITE_JPEG_QUALITY), 10]  # Lower quality = more compression
result, encodedimage = cv2.imencode('.jpg', image, encoding_parameter)

# Decode back to simulate JPEG transformation
converted = cv2.imdecode(encodedimage, 1)
converted_rgb = cv2.cvtColor(converted, cv2.COLOR_BGR2RGB)

#Print
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(converted_rgb)
plt.title('JPEG Transformed (Quality=10)')
plt.axis('off')

plt.tight_layout()
plt.show()
