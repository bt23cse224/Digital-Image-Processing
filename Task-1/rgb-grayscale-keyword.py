import cv2
image = cv2.imread('primary-image.jpg')

# Convert BGR image to Grayscale using OpenCV color conversion keyword
gyimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayscale-image.jpg', gyimg)
cv2.imshow('Grayscale Image', gyimg)

cv2.waitKey(0)
cv2.destroyAllWindows()
