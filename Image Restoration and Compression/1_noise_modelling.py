import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

img = cv2.imread("image.jpg", 0)

rows, cols = img.shape

# ----------------------------------
# Gaussian Noise (Box-Muller)
# ----------------------------------

gaussian = img.copy().astype(float)

mean = 0
sigma = 25

for i in range(rows):
    for j in range(cols):

        u1 = random.random()
        u2 = random.random()

        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

        noise = mean + sigma * z

        pixel = img[i][j] + noise

        if pixel < 0:
            pixel = 0

        if pixel > 255:
            pixel = 255

        gaussian[i][j] = pixel

gaussian = gaussian.astype(np.uint8)

# ----------------------------------
# Rayleigh Noise
# ----------------------------------

rayleigh = img.copy().astype(float)

sigma_r = 20

for i in range(rows):
    for j in range(cols):

        u = random.random()

        noise = sigma_r * math.sqrt(-2 * math.log(1 - u))

        pixel = img[i][j] + noise

        if pixel > 255:
            pixel = 255

        rayleigh[i][j] = pixel

rayleigh = rayleigh.astype(np.uint8)

# ----------------------------------
# Salt & Pepper Noise
# ----------------------------------

sp = img.copy()

prob = 0.05

for i in range(rows):
    for j in range(cols):

        r = random.random()

        if r < prob/2:
            sp[i][j] = 0

        elif r < prob:
            sp[i][j] = 255

# ----------------------------------
# Display
# ----------------------------------

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img,cmap='gray')
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(gaussian,cmap='gray')
plt.title("Gaussian Noise")

plt.subplot(2,2,3)
plt.imshow(rayleigh,cmap='gray')
plt.title("Rayleigh Noise")

plt.subplot(2,2,4)
plt.imshow(sp,cmap='gray')
plt.title("Salt & Pepper Noise")

plt.show()
