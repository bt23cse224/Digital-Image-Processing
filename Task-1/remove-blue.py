import cv2
import numpy as np

# opencv loads images in BGR format only
image_bgr = cv2.imread('primary-image.jpg')

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Set value at blue index to 0
newimg = image_rgb.copy()
newimg[:, :, 2] = 0 

# Convert back to BGR for OpenCV display
newimg = cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR)
cv2.imwrite('img-without-blue.jpg', newimg)

cv2.imshow('Image Without Blue', newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
