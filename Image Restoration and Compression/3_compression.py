import cv2
import numpy as np
import math

# ==================================
# HUFFMAN CODING
# ==================================

text = "DIGITALIMAGEPROCESSING"

freq = {}

for ch in text:

    if ch in freq:
        freq[ch] += 1
    else:
        freq[ch] = 1

print("Frequency Table")

for k in freq:
    print(k, freq[k])

# ==================================
# DCT COMPRESSION
# ==================================

img = cv2.imread("image.jpg",0)

block = img[0:8,0:8].astype(float)

DCT = np.zeros((8,8))

for u in range(8):

    for v in range(8):

        if u == 0:
            cu = 1/math.sqrt(2)
        else:
            cu = 1

        if v == 0:
            cv = 1/math.sqrt(2)
        else:
            cv = 1

        total = 0

        for x in range(8):

            for y in range(8):

                total += (
                    block[x][y]
                    *
                    math.cos((2*x+1)*u*math.pi/16)
                    *
                    math.cos((2*y+1)*v*math.pi/16)
                )

        DCT[u][v] = 0.25 * cu * cv * total

print("\nDCT Matrix\n")
print(DCT)

# ==================================
# Compression
# Keep only low frequencies
# ==================================

compressed = np.zeros((8,8))

for i in range(4):
    for j in range(4):

        compressed[i][j] = DCT[i][j]

print("\nCompressed DCT Matrix\n")
print(compressed)
