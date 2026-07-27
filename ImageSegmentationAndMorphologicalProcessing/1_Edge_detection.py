import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Read Image
# -------------------------

img = cv2.imread('image.jpg', 0)

rows, cols = img.shape

# -------------------------
# SOBEL OPERATOR
# -------------------------

sobel_x = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

sobel_y = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]

sobel_output = np.zeros((rows, cols), dtype=np.uint8)

for i in range(1, rows-1):
    for j in range(1, cols-1):

        gx = 0
        gy = 0

        for m in range(-1, 2):
            for n in range(-1, 2):

                gx += img[i+m, j+n] * sobel_x[m+1][n+1]
                gy += img[i+m, j+n] * sobel_y[m+1][n+1]

        magnitude = int((gx**2 + gy**2)**0.5)

        if magnitude > 255:
            magnitude = 255

        sobel_output[i, j] = magnitude

# -------------------------
# PREWITT OPERATOR
# -------------------------

prewitt_x = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
]

prewitt_y = [
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
]

prewitt_output = np.zeros((rows, cols), dtype=np.uint8)

for i in range(1, rows-1):
    for j in range(1, cols-1):

        gx = 0
        gy = 0

        for m in range(-1, 2):
            for n in range(-1, 2):

                gx += img[i+m, j+n] * prewitt_x[m+1][n+1]
                gy += img[i+m, j+n] * prewitt_y[m+1][n+1]

        magnitude = int((gx**2 + gy**2)**0.5)

        if magnitude > 255:
            magnitude = 255

        prewitt_output[i, j] = magnitude

# -------------------------
# ROBERTS CROSS OPERATOR
# -------------------------

roberts_x = [
    [1, 0],
    [0,-1]
]

roberts_y = [
    [0, 1],
    [-1,0]
]

roberts_output = np.zeros((rows, cols), dtype=np.uint8)

for i in range(rows-1):
    for j in range(cols-1):

        gx = 0
        gy = 0

        for m in range(2):
            for n in range(2):

                gx += img[i+m, j+n] * roberts_x[m][n]
                gy += img[i+m, j+n] * roberts_y[m][n]

        magnitude = int((gx**2 + gy**2)**0.5)

        if magnitude > 255:
            magnitude = 255

        roberts_output[i, j] = magnitude

# -------------------------
# DISPLAY RESULTS
# -------------------------

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(sobel_output, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(prewitt_output, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(roberts_output, cmap='gray')
plt.title("Roberts Cross Edge Detection")
plt.axis('off')

plt.show()
