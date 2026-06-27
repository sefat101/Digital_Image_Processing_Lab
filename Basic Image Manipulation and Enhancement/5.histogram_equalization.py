import cv2
import matplotlib.pyplot as plt

# Read the image in grayscale
img = cv2.imread("meow.jpeg", 0)

# Check if image is loaded
if img is None:
    print("Error: Image not found!")
    exit()

# Histogram Equalization
equalized = cv2.equalizeHist(img)

# Create a 2x2 subplot
plt.figure(figsize=(10, 8))

# 1. Input Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis("off")

# 2. Output Image
plt.subplot(2, 2, 2)
plt.imshow(equalized, cmap='gray')
plt.title("Output Image")
plt.axis("off")

# 3. Histogram Before Processing
plt.subplot(2, 2, 3)
plt.hist(img.ravel(), bins=256, range=[0, 256], color='blue')
plt.title("Histogram Before Processing")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# 4. Histogram After Processing
plt.subplot(2, 2, 4)
plt.hist(equalized.ravel(), bins=256, range=[0, 256], color='green')
plt.title("Histogram After Processing")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
