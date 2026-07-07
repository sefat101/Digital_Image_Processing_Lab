import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Read Images
# -------------------------

source = cv2.imread('source.jpg', 0)
reference = cv2.imread('reference.jpg', 0)

rows, cols = source.shape

# -------------------------
# Step 1: Source Histogram
# -------------------------

hist_source = [0] * 256

for i in range(rows):
    for j in range(cols):

        pixel = source[i, j]
        hist_source[pixel] += 1

# -------------------------
# Step 2: Reference Histogram
# -------------------------

hist_reference = [0] * 256

r_rows, r_cols = reference.shape

for i in range(r_rows):
    for j in range(r_cols):

        pixel = reference[i, j]
        hist_reference[pixel] += 1

# -------------------------
# Step 3: PDF
# -------------------------

pdf_source = [0] * 256
pdf_reference = [0] * 256

total_source = rows * cols
total_reference = r_rows * r_cols

for i in range(256):

    pdf_source[i] = hist_source[i] / total_source
    pdf_reference[i] = hist_reference[i] / total_reference

# -------------------------
# Step 4: CDF
# -------------------------

cdf_source = [0] * 256
cdf_reference = [0] * 256

cdf_source[0] = pdf_source[0]
cdf_reference[0] = pdf_reference[0]

for i in range(1, 256):

    cdf_source[i] = cdf_source[i-1] + pdf_source[i]
    cdf_reference[i] = cdf_reference[i-1] + pdf_reference[i]

# -------------------------
# Step 5: Mapping
# -------------------------

mapping = [0] * 256

for i in range(256):

    diff = abs(cdf_source[i] - cdf_reference[0])
    best_match = 0

    for j in range(256):

        new_diff = abs(cdf_source[i] - cdf_reference[j])

        if new_diff < diff:
            diff = new_diff
            best_match = j

    mapping[i] = best_match

# -------------------------
# Step 6: Generate Output
# -------------------------

output = np.zeros((rows, cols), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):

        output[i, j] = mapping[source[i, j]]

# -------------------------
# Display Images
# -------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1,3,1)
plt.imshow(source, cmap='gray')
plt.title("Source Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(reference, cmap='gray')
plt.title("Reference Image")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(output, cmap='gray')
plt.title("Histogram Specified")
plt.axis('off')

plt.show()
