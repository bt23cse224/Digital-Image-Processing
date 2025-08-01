import cv2
import numpy as np

img = cv2.imread('primary-image.jpg')
rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = (0.2989 * rgbimg[:, :, 0] + 0.5870 * rgbimg[:, :, 1] + 0.1140 * rgbimg[:, :, 2]).astype(np.uint8)
cv2.imwrite('gray-using-avg.jpg', gray)

cv2.imshow('Gray Luminosity', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
