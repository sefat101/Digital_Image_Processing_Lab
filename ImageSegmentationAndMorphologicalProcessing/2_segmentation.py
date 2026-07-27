import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("image.jpg", 0)

rows, cols = img.shape

# -----------------------------
# POINT DETECTION
# -----------------------------

point_kernel = [
    [-1,-1,-1],
    [-1, 8,-1],
    [-1,-1,-1]
]

point_img = np.zeros((rows, cols), dtype=np.uint8)

for i in range(1, rows-1):
    for j in range(1, cols-1):

        value = 0

        for m in range(-1,2):
            for n in range(-1,2):

                value += img[i+m,j+n] * point_kernel[m+1][n+1]

        value = abs(value)

        if value > 255:
            value = 255

        point_img[i,j] = value

# -----------------------------
# LINE DETECTION (HORIZONTAL)
# -----------------------------

line_kernel = [
    [-1,-1,-1],
    [ 2, 2, 2],
    [-1,-1,-1]
]

line_img = np.zeros((rows, cols), dtype=np.uint8)

for i in range(1, rows-1):
    for j in range(1, cols-1):

        value = 0

        for m in range(-1,2):
            for n in range(-1,2):

                value += img[i+m,j+n] * line_kernel[m+1][n+1]

        value = abs(value)

        if value > 255:
            value = 255

        line_img[i,j] = value

# -----------------------------
# EDGE DETECTION (SOBEL)
# -----------------------------

gx_kernel = [
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]

gy_kernel = [
    [-1,-2,-1],
    [ 0, 0, 0],
    [ 1, 2, 1]
]

edge_img = np.zeros((rows, cols), dtype=np.uint8)

for i in range(1, rows-1):
    for j in range(1, cols-1):

        gx = 0
        gy = 0

        for m in range(-1,2):
            for n in range(-1,2):

                gx += img[i+m,j+n] * gx_kernel[m+1][n+1]
                gy += img[i+m,j+n] * gy_kernel[m+1][n+1]

        magnitude = int((gx**2 + gy**2)**0.5)

        if magnitude > 255:
            magnitude = 255

        edge_img[i,j] = magnitude

# -----------------------------
# THRESHOLDING
# -----------------------------

threshold = 128

binary_img = np.zeros((rows, cols), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):

        if img[i,j] > threshold:
            binary_img[i,j] = 255
        else:
            binary_img[i,j] = 0

# -----------------------------
# DISPLAY
# -----------------------------

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img,cmap='gray')
plt.title("Original")

plt.subplot(2,3,2)
plt.imshow(point_img,cmap='gray')
plt.title("Point Detection")

plt.subplot(2,3,3)
plt.imshow(line_img,cmap='gray')
plt.title("Line Detection")

plt.subplot(2,3,4)
plt.imshow(edge_img,cmap='gray')
plt.title("Edge Detection")

plt.subplot(2,3,5)
plt.imshow(binary_img,cmap='gray')
plt.title("Thresholding")

plt.show()
