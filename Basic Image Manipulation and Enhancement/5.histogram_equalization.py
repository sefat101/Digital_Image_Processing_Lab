import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("meow.jpeg", 0)

if img is None:
    print("Image not found!")
    exit()

# Image size
rows, cols = img.shape
total_pixels = rows * cols

# -----------------------------
# Step 1: Calculate Histogram
# -----------------------------
hist = np.zeros(256, dtype=int)

for i in range(rows):
    for j in range(cols):
        hist[img[i, j]] += 1

# -----------------------------
# Step 2: Calculate PDF
# -----------------------------
pdf = hist / total_pixels

# -----------------------------
# Step 3: Calculate CDF
# -----------------------------
cdf = np.zeros(256)

cdf[0] = pdf[0]

for i in range(1, 256):
    cdf[i] = cdf[i - 1] + pdf[i]

# -----------------------------
# Step 4: Create Transformation
# -----------------------------
transform = np.round(cdf * 255).astype(np.uint8)

# -----------------------------
# Step 5: Generate Equalized Image
# -----------------------------
equalized = np.zeros_like(img)

for i in range(rows):
    for j in range(cols):
        equalized[i, j] = transform[img[i, j]]

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(10, 8))

# Input Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis("off")

# Output Image
plt.subplot(2, 2, 2)
plt.imshow(equalized, cmap='gray')
plt.title("Equalized Image")
plt.axis("off")

# Histogram Before
plt.subplot(2, 2, 3)
plt.bar(range(256), hist, color='blue')
plt.title("Histogram Before")
plt.xlabel("Gray Level")
plt.ylabel("Frequency")

# Histogram After
hist_eq = np.zeros(256, dtype=int)

for i in range(rows):
    for j in range(cols):
        hist_eq[equalized[i, j]] += 1

plt.subplot(2, 2, 4)
plt.bar(range(256), hist_eq, color='green')
plt.title("Histogram After")
plt.xlabel("Gray Level")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
