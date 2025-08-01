import cv2
import numpy as np

gray = cv2.imread('grayscale-image.jpg', cv2.IMREAD_GRAYSCALE)

height, width = gray.shape 

bw = np.zeros_like(gray)

for i in range(height):
    for j in range(width):
        if gray[i, j] >= 128:
            bw[i, j] = 255
        else:
            bw[i, j] = 0




cv2.imwrite('gray-bw-midvalue.jpg', bw)
cv2.imshow('Black and White', bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
