import cv2

image = cv2.imread('primary-image.jpg')

if image is None:
    print("Could not read the image.")
else:
    cv2.imshow('Image Window', image)
    # Wait until any key is pressed
    cv2.waitKey(0)
    # Close all OpenCV windows
    cv2.destroyAllWindows()
