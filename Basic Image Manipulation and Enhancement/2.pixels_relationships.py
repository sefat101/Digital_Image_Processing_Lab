#implementation of relationships of pixel in one image 5*5


import numpy as np

img = np.array([
    [1,0,1,0,1],
    [0,1,1,0,0],
    [1,1,0,1,0],
    [0,0,1,1,1],
    [1,0,0,1,0]
])

r, c = 2, 2
source_val = img[r, c]

# Extract the local 3x3 block around (r, c)
# This gets rows r-1 to r+1 and columns c-1 to c+1 safely
window = img[r-1:r+2, c-1:c+2]

print(f"Source Element (S) at ({r},{c}) has value: {source_val}\n")

# --- 1. 4-NEIGHBORHOOD MATRIX ---
print("### 1. 4-Neighborhood Matrix")
print(f"    [{window[0, 1]}]    ")
print(f"[{window[1, 0]}]  S  [{window[1, 2]}]")
print(f"    [{window[2, 1]}]    ")

print("\n" + "="*30 + "\n")

# --- 2. DIAGONAL NEIGHBORHOOD MATRIX ---
print("### 2. Diagonal Neighborhood Matrix")
print(f"[{window[0, 0]}]     [{window[0, 2]}]")
print(f"     S     ")
print(f"[{window[2, 0]}]     [{window[2, 2]}]")

print("\n" + "="*30 + "\n")

# --- 3. 8-NEIGHBORHOOD MATRIX ---
print("### 3. 8-Neighborhood Matrix")
print(f"[{window[0, 0]}] [{window[0, 1]}] [{window[0, 2]}]")
print(f"[{window[1, 0]}]  S  [{window[1, 2]}]")
print(f"[{window[2, 0]}] [{window[2, 1]}] [{window[2, 2]}]")
