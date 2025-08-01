import cv2
import numpy as np

# Read the image in BGR format
imgbgr = cv2.imread('primary-image.jpg')

# Convert BGR to RGB
imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)

# Calculate average manually: sum channels then divide by 3
gyavg = ((imgrgb[:, :, 0].astype(int) + imgrgb[:, :, 1].astype(int) + imgrgb[:, :, 2].astype(int)) // 3).astype(np.uint8)
cv2.imwrite('gray-using-avg.jpg', gyavg)


cv2.imshow('Grayscale (Average)', gyavg)
cv2.waitKey(0)
cv2.destroyAllWindows()
