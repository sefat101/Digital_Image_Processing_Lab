import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("noisy.jpg",0)

rows, cols = img.shape

# ----------------------------------
# Mean Filter
# ----------------------------------

mean_restored = np.zeros((rows,cols),dtype=np.uint8)

for i in range(1,rows-1):
    for j in range(1,cols-1):

        total = 0

        for m in range(-1,2):
            for n in range(-1,2):

                total += int(img[i+m][j+n])

        mean_restored[i][j] = total // 9

# ----------------------------------
# Median Filter
# ----------------------------------

median_restored = np.zeros((rows,cols),dtype=np.uint8)

for i in range(1,rows-1):
    for j in range(1,cols-1):

        arr = []

        for m in range(-1,2):
            for n in range(-1,2):

                arr.append(int(img[i+m][j+n]))

        # Bubble Sort

        length = len(arr)

        for x in range(length):

            for y in range(length-x-1):

                if arr[y] > arr[y+1]:

                    temp = arr[y]
                    arr[y] = arr[y+1]
                    arr[y+1] = temp

        median_restored[i][j] = arr[4]

# ----------------------------------
# Display
# ----------------------------------

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.title("Noisy")

plt.subplot(1,3,2)
plt.imshow(mean_restored,cmap='gray')
plt.title("Mean Filter")

plt.subplot(1,3,3)
plt.imshow(median_restored,cmap='gray')
plt.title("Median Filter")

plt.show()
