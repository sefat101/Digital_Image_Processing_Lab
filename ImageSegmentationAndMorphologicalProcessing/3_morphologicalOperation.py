import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read binary image
img = cv2.imread("binary.jpg", 0)

rows, cols = img.shape

# Convert to binary

for i in range(rows):
    for j in range(cols):

        if img[i,j] > 128:
            img[i,j] = 255
        else:
            img[i,j] = 0

# -----------------------------
# DILATION
# -----------------------------

dilation = np.zeros((rows,cols),dtype=np.uint8)

for i in range(1,rows-1):
    for j in range(1,cols-1):

        found = False

        for m in range(-1,2):
            for n in range(-1,2):

                if img[i+m,j+n] == 255:
                    found = True

        if found:
            dilation[i,j] = 255

# -----------------------------
# EROSION
# -----------------------------

erosion = np.zeros((rows,cols),dtype=np.uint8)

for i in range(1,rows-1):
    for j in range(1,cols-1):

        keep = True

        for m in range(-1,2):
            for n in range(-1,2):

                if img[i+m,j+n] == 0:
                    keep = False

        if keep:
            erosion[i,j] = 255

# -----------------------------
# OPENING = EROSION + DILATION
# -----------------------------

opening = np.zeros((rows,cols),dtype=np.uint8)

for i in range(1,rows-1):
    for j in range(1,cols-1):

        found = False

        for m in range(-1,2):
            for n in range(-1,2):

                if erosion[i+m,j+n] == 255:
                    found = True

        if found:
            opening[i,j] = 255

# -----------------------------
# CLOSING = DILATION + EROSION
# -----------------------------

closing = np.zeros((rows,cols),dtype=np.uint8)

for i in range(1,rows-1):
    for j in range(1,cols-1):

        keep = True

        for m in range(-1,2):
            for n in range(-1,2):

                if dilation[i+m,j+n] == 0:
                    keep = False

        if keep:
            closing[i,j] = 255

# -----------------------------
# TOP HAT
# -----------------------------

top_hat = img.astype(np.int32) - opening.astype(np.int32)

for i in range(rows):
    for j in range(cols):

        if top_hat[i,j] < 0:
            top_hat[i,j] = 0

top_hat = top_hat.astype(np.uint8)

# -----------------------------
# DISPLAY
# -----------------------------

plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.imshow(img,cmap='gray')
plt.title("Original")

plt.subplot(2,3,2)
plt.imshow(dilation,cmap='gray')
plt.title("Dilation")

plt.subplot(2,3,3)
plt.imshow(erosion,cmap='gray')
plt.title("Erosion")

plt.subplot(2,3,4)
plt.imshow(opening,cmap='gray')
plt.title("Opening")

plt.subplot(2,3,5)
plt.imshow(closing,cmap='gray')
plt.title("Closing")

plt.subplot(2,3,6)
plt.imshow(top_hat,cmap='gray')
plt.title("Top Hat")

plt.show()
