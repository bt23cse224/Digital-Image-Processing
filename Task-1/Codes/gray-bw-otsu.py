import cv2

gray = cv2.imread('grayscale-image.jpg', cv2.IMREAD_GRAYSCALE)

threshold_value = 0  # Placeholder, ignored by Otsu
max_value = 255      # Value for white pixels

# The function returns two values-the chosen threshold value and the binary image
ret, bw = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("Otsu's chosen threshold:", ret)
cv2.imwrite('gray-bw-otsu.jpg', bw)

cv2.imshow('Black and White (Otsu)', bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
