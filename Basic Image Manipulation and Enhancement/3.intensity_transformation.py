import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('image.jpg', 0)

# -------------------------------
# 1. Image Negative
# -------------------------------
negative = 255 - img

# -------------------------------
# 2. Log Transformation
# -------------------------------
c = 255 / np.log(1 + np.max(img))

log_transformed = c * np.log(1 + img)

log_transformed = np.array(log_transformed,
                           dtype=np.uint8)

# -------------------------------
# 3. Gamma Transformation
#pip install opencv-python
# -------------------------------

gamma = 0.5

normalized = img / 255.0

gamma_corrected = np.power(normalized, gamma)

gamma_corrected = np.uint8(gamma_corrected * 255)

# -------------------------------
# Display
# -------------------------------

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(negative, cmap='gray')
plt.title("Negative")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(log_transformed, cmap='gray')
plt.title("Log Transformation")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(gamma_corrected, cmap='gray')
plt.title("Gamma Transformation")
plt.axis('off')

plt.show()
