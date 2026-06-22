import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# TASK 1: Read Grayscale Image
# =====================================================

image = cv2.imread("meow.jpeg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found!")
    exit()

print("Image Loaded Successfully")
print("Image Shape:", image.shape)

# =====================================================
# TASK 2: Convert Each Pixel to Binary Representation
# =====================================================

binary_image = np.vectorize(lambda pixel: format(pixel, '08b'))(image)

# Save binary representation into a text file
np.savetxt(
    "binary_representation.txt",
    binary_image,
    fmt="%s"
)

print("Binary representation saved as binary_representation.txt")

# =====================================================
# TASK 3: Extract All 8 Bit Planes
# =====================================================

bit_planes = []

for bit in range(8):
    plane = ((image >> bit) & 1) * 255
    bit_planes.append(plane.astype(np.uint8))

print("All 8 bit planes extracted successfully.")

# =====================================================
# TASK 4: Save Individual Bit Planes
# =====================================================

for i in range(8):
    cv2.imwrite(f"bit_plane_{i}.png", bit_planes[i])

print("All bit planes saved successfully.")

# =====================================================
# TASK 5: MSB and LSB Analysis
# =====================================================

msb = bit_planes[7]
lsb = bit_planes[0]

cv2.imwrite("MSB_BitPlane7.png", msb)
cv2.imwrite("LSB_BitPlane0.png", lsb)

# =====================================================
# TASK 6: Reconstruct Compressed Image
# Keeping Higher Order Bit Planes:
# Bit 7, Bit 6, Bit 5, Bit 4
# =====================================================

compressed = np.zeros_like(image)

for bit in range(4, 8):
    compressed += (((image >> bit) & 1) << bit)

compressed = compressed.astype(np.uint8)

cv2.imwrite("compressed_image.png", compressed)

print("Compressed image created successfully.")

# =====================================================
# TASK 7: Compression Analysis
# =====================================================

original_bits = 8
compressed_bits = 4

compression_ratio = original_bits / compressed_bits

print("\nCompression Analysis")
print("------------------------------")
print("Original Bit Depth   :", original_bits, "bits")
print("Compressed Bit Depth :", compressed_bits, "bits")
print(f"Compression Ratio    : {compression_ratio:.2f}:1")

# =====================================================
# TASK 8: Display Results
# =====================================================

plt.figure(figsize=(15, 12))

# Original Image
plt.subplot(3, 4, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.text(
    0.5, -0.15,
    "Original Image",
    fontsize=12,
    ha='center',
    va='top',
    transform=plt.gca().transAxes
)

# Bit Planes 0-7
for i in range(8):
    plt.subplot(3, 4, i + 2)
    plt.imshow(bit_planes[i], cmap='gray')
    plt.axis('off')
    plt.text(
        0.5, -0.15,
        f"Bit Plane {i}",
        fontsize=12,
        ha='center',
        va='top',
        transform=plt.gca().transAxes
    )

# Compressed Image
plt.subplot(3, 4, 10)
plt.imshow(compressed, cmap='gray')
plt.axis('off')
plt.text(
    0.5, -0.15,
    "Compressed Image",
    fontsize=12,
    ha='center',
    va='top',
    transform=plt.gca().transAxes
)

# MSB
plt.subplot(3, 4, 11)
plt.imshow(bit_planes[7], cmap='gray')
plt.axis('off')
plt.text(
    0.5, -0.15,
    "MSB (Bit Plane 7)",
    fontsize=12,
    ha='center',
    va='top',
    transform=plt.gca().transAxes
)

# LSB
plt.subplot(3, 4, 12)
plt.imshow(bit_planes[0], cmap='gray')
plt.axis('off')
plt.text(
    0.5, -0.15,
    "LSB (Bit Plane 0)",
    fontsize=12,
    ha='center',
    va='top',
    transform=plt.gca().transAxes
)

# Layout adjustments
plt.tight_layout()
plt.subplots_adjust(hspace=0.6, bottom=0.08)

# Save Figure
plt.savefig(
    "bit_plane_slicing_result.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
