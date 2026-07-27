import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("color.jpg")

rows, cols, _ = img.shape

# -----------------------------
# RGB EXTRACTION
# -----------------------------

B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]

# -----------------------------
# CMY CONVERSION
# -----------------------------

C = 255 - R
M = 255 - G
Y = 255 - B

# -----------------------------
# MANUAL HSV CONVERSION
# -----------------------------

H = np.zeros((rows,cols))
S = np.zeros((rows,cols))
V = np.zeros((rows,cols))

for i in range(rows):
    for j in range(cols):

        r = R[i,j] / 255.0
        g = G[i,j] / 255.0
        b = B[i,j] / 255.0

        cmax = max(r,g,b)
        cmin = min(r,g,b)

        delta = cmax - cmin

        # Hue

        if delta == 0:
            h = 0

        elif cmax == r:
            h = 60 * (((g-b)/delta) % 6)

        elif cmax == g:
            h = 60 * (((b-r)/delta) + 2)

        else:
            h = 60 * (((r-g)/delta) + 4)

        # Saturation

        if cmax == 0:
            s = 0
        else:
            s = delta / cmax

        # Value

        v = cmax

        H[i,j] = h
        S[i,j] = s
        V[i,j] = v

# Normalize for display

H_display = (H/360*255).astype(np.uint8)
S_display = (S*255).astype(np.uint8)
V_display = (V*255).astype(np.uint8)

# -----------------------------
# DISPLAY
# -----------------------------

plt.figure(figsize=(15,10))

plt.subplot(3,4,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(3,4,2)
plt.imshow(R,cmap='gray')
plt.title("Red")

plt.subplot(3,4,3)
plt.imshow(G,cmap='gray')
plt.title("Green")

plt.subplot(3,4,4)
plt.imshow(B,cmap='gray')
plt.title("Blue")

plt.subplot(3,4,5)
plt.imshow(C,cmap='gray')
plt.title("Cyan")

plt.subplot(3,4,6)
plt.imshow(M,cmap='gray')
plt.title("Magenta")

plt.subplot(3,4,7)
plt.imshow(Y,cmap='gray')
plt.title("Yellow")

plt.subplot(3,4,8)
plt.imshow(H_display,cmap='gray')
plt.title("Hue")

plt.subplot(3,4,9)
plt.imshow(S_display,cmap='gray')
plt.title("Saturation")

plt.subplot(3,4,10)
plt.imshow(V_display,cmap='gray')
plt.title("Value")

plt.show()
