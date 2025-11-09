import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1. Load sample digits dataset (samples of 8x8 images)
digits = load_digits()
images = digits.images
labels = digits.target

# 2. Flatten images for knn input (each 8x8 = 64 features)
data = images.reshape((len(images), -1)).astype(np.float32)

# 3. Split data into training and testing sets
trainData, testData, trainLabels, testLabels = train_test_split(
    data, labels, test_size=0.3, random_state=42
)

# 4. Create and train model
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels.astype(np.float32))

# 5. Predict using KNN (k=5)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)

# 6. Calculate accuracy
matches = result.ravel() == testLabels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size

print(f'Accuracy: {accuracy:.2f}%')

# 7. Test samples with predictions
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(testData[i].reshape(8, 8), cmap='gray')
    plt.title(f'Pred: {int(result[i])}')
    plt.axis('off')
plt.tight_layout()
plt.show()
