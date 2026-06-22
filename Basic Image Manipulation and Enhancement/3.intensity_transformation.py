#pip install opencv-python


import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Read Grayscale Image
# =====================================================

image = cv2.imread('meow.jpeg', cv2.IMREAD_GRAYSCALE)


if image is None:
    print("Error: Image not found!")
    exit()

# =====================================================
# 1. Negative Transformation
# =====================================================

negative = 255 - image

# =====================================================
# 2. Log Transformation (Fixed Overflow)
# =====================================================

# 1. Convert the image to float32 to prevent uint8 overflow (255 + 1 = 256, not 0)
float_image = image.astype(np.float32)

# 2. Safely calculate the scaling constant 'c'
c = 255 / np.log(1 + np.max(float_image))

# 3. Apply the log transformation formula
log_image = c * np.log(1 + float_image)

# 4. Safely clip and convert back to uint8 for displaying/saving
log_image = np.clip(log_image, 0, 255).astype(np.uint8)

# =====================================================
# 3. Gamma Transformation
# =====================================================

gamma = 0.5

normalized_image = image / 255.0

gamma_image = np.power(normalized_image, gamma)

gamma_image = np.uint8(gamma_image * 255)

# =====================================================
# 4. Contrast Stretching
# =====================================================

r_min = np.min(image)
r_max = np.max(image)

contrast_image = ((image - r_min) /
                  (r_max - r_min)) * 255

contrast_image = np.uint8(contrast_image)

# =====================================================
# Save Results
# =====================================================

cv2.imwrite("negative.jpg", negative)
cv2.imwrite("log.jpg", log_image)
cv2.imwrite("gamma.jpg", gamma_image)
cv2.imwrite("contrast.jpg", contrast_image)

# =====================================================
# Display Results
# =====================================================

# A wider, shorter figure ratio works perfectly for this layout
plt.figure(figsize=(16, 8))

# --- ROW 1: Original Image (Spans the full top row) ---
# We treat the top area as a 2-row, 1-column grid, and place this in position 1
plt.subplot(2, 1, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image", fontsize=14, fontweight='bold')
plt.axis('off')

# --- ROW 2: Transformations (4 images side-by-side) ---
# Now we treat the layout as a 2-row, 4-column grid. 
# The second row slots correspond to indices 5, 6, 7, and 8.

# Position 5 (Row 2, Column 1)
plt.subplot(2, 4, 5)
plt.imshow(negative, cmap='gray')
plt.title("Negative", fontsize=12)
plt.axis('off')

# Position 6 (Row 2, Column 2)
plt.subplot(2, 4, 6)
plt.imshow(log_image, cmap='gray')
plt.title("Log Transformation", fontsize=12)
plt.axis('off')

# Position 7 (Row 2, Column 3)
plt.subplot(2, 4, 7)
plt.imshow(gamma_image,cmap='gray' )
plt.title("Gamma Transformation", fontsize=12)
plt.axis('off')

# Position 8 (Row 2, Column 4)
plt.subplot(2, 4, 8)
plt.imshow(contrast_image, cmap='gray')
plt.title("Contrast Stretching", fontsize=12)
plt.axis('off')

# Clean up spacing so layouts don't overlap
plt.tight_layout()
plt.savefig("transformation_layout_results.png", dpi=300, bbox_inches="tight")
plt.show()

print("All transformed images saved successfully.")
