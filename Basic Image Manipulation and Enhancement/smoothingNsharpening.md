
# Lab 1: Smoothing Filters (Mean Filter + Median Filter)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
img = cv2.imread('image.jpg', 0)

rows, cols = img.shape

# -------------------------
# Mean Filter (3x3)
# -------------------------

mean_filtered = np.zeros((rows, cols), dtype=np.uint8)

for i in range(1, rows-1):
    for j in range(1, cols-1):

        total = 0

        for m in range(-1, 2):
            for n in range(-1, 2):
                total += img[i+m, j+n]

        mean_filtered[i, j] = total // 9

# -------------------------
# Median Filter (3x3)
# -------------------------

median_filtered = np.zeros((rows, cols), dtype=np.uint8)

for i in range(1, rows-1):
    for j in range(1, cols-1):

        neighbors = []

        for m in range(-1, 2):
            for n in range(-1, 2):
                neighbors.append(img[i+m, j+n])

        neighbors.sort()

        median_filtered[i, j] = neighbors[4]

# -------------------------
# Display
# -------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(mean_filtered, cmap='gray')
plt.title("Mean Filter")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(median_filtered, cmap='gray')
plt.title("Median Filter")
plt.axis('off')

plt.show()
```

---

# Mean Filter Theory

Kernel:

[
\frac{1}{9}
\begin{bmatrix}
1&1&1\
1&1&1\
1&1&1
\end{bmatrix}
]

Example:

```text
10 20 30
40 50 60
70 80 90
```

Output:

[
\frac{10+20+30+40+50+60+70+80+90}{9}
]

[
=50
]

---

# Median Filter Theory

Example:

```text
10 20 30
40 200 60
70 80 90
```

Sort:

```text
10 20 30 40 60 70 80 90 200
```

Median:

```text
60
```

Salt-and-pepper noise removal-এর জন্য Median Filter সবচেয়ে কার্যকর।

---

# Lab 2: Sharpening Filters (Laplacian + High Boost)

## Laplacian Filter (Manual)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)

rows, cols = img.shape

laplacian = np.zeros((rows, cols), dtype=np.int32)

kernel = [
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
]

for i in range(1, rows-1):
    for j in range(1, cols-1):

        value = 0

        for m in range(-1, 2):
            for n in range(-1, 2):

                value += img[i+m, j+n] * kernel[m+1][n+1]

        laplacian[i, j] = value

laplacian = np.clip(laplacian, 0, 255)

laplacian = laplacian.astype(np.uint8)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Filter")

plt.show()
```

---

# Laplacian Kernel

[
\begin{bmatrix}
0 & -1 & 0\
-1 & 4 & -1\
0 & -1 & 0
\end{bmatrix}
]

Purpose:

* Edge Enhancement
* Image Sharpening

---

# High Boost Filtering (Manual)

Formula:

[
Mask = Original - Blurred
]

[
HighBoost = Original + k(Mask)
]

where

[
k > 1
]

---

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)

rows, cols = img.shape

# -------------------------
# Mean Blur Manually
# -------------------------

blurred = np.zeros((rows, cols), dtype=np.uint8)

for i in range(1, rows-1):
    for j in range(1, cols-1):

        total = 0

        for m in range(-1, 2):
            for n in range(-1, 2):

                total += img[i+m, j+n]

        blurred[i, j] = total // 9

# -------------------------
# Unsharp Mask
# -------------------------

mask = img.astype(np.int32) - blurred.astype(np.int32)

# -------------------------
# High Boost
# -------------------------

k = 2

high_boost = img.astype(np.int32) + k * mask

high_boost = np.clip(high_boost, 0, 255)

high_boost = high_boost.astype(np.uint8)

# -------------------------
# Display
# -------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(mask, cmap='gray')
plt.title("Mask")

plt.subplot(1,3,3)
plt.imshow(high_boost, cmap='gray')
plt.title("High Boost")

plt.show()
```

# Viva Questions

### Why Mean Filter?

Removes random noise by averaging neighboring pixels.

### Why Median Filter?

Removes salt-and-pepper noise while preserving edges.

### Why Laplacian Filter?

Uses second-order derivatives to highlight edges and fine details.

### Why High Boost Filtering?

Enhances edges while retaining the original image information.

### Difference Between Unsharp Masking and High Boost?

* Unsharp Masking → (k=1)
* High Boost Filtering → (k>1)

