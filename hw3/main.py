'''
Author: Cody Costa
Date:   8/2/2025

'''

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.fft import dctn, idctn
from scipy.linalg import hadamard


''' PROBLEM 1:  DCT/IDCT 4X4 '''

# starting matrix
matrix = np.array([[1, 0, 1, 0],
                   [2, 0, 2, 0],
                   [0, 1, 0, 1],
                   [-1, 0, -1, 0]])

# compute DCT
dct_matrix = dctn(matrix, type=2, norm='ortho')

# compute IDCT
idct_matrix = idctn(matrix, type=2, norm='ortho')

# print results
print(f'DCT:\n{dct_matrix}\n\nIDCT:\n{idct_matrix}')


''' PROBLEM 2:  BASIS FUNCTIONS OF DCT '''

def dct_basis(N):
    basis = []
    for u in range(N):
        for v in range(N):
            coeff = np.zeros((N, N))
            coeff[u, v] = 1  # Impulse at (u, v)
            
            # 2D inverse DCT with orthogonal normalization
            basis_fn = idctn(coeff, norm='ortho')
            basis.append(basis_fn)
    return basis


N = 4
basis_funcs = dct_basis(N)

# # Plot 16 basis functions
fig, axes = plt.subplots(N, N, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(basis_funcs[i], cmap='gray')
    ax.axis('off')

    # Add border (rectangle)
    rect = patches.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        linewidth=1.5, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

plt.tight_layout()
plt.show()


''' PROBLEM 3:  WALSH HADAMARD TRANSFORM '''

def sequency_order(H):
    """Reorder Hadamard matrix rows to sequency order."""
    n = H.shape[0]
    # Count sign changes in each row
    sign_changes = [np.sum(H[i, :-1] != H[i, 1:]) for i in range(n)]
    order = np.argsort(sign_changes)  # Sort by sign changes
    return H[order]

# Generate 4x4 Hadamard matrix
H = hadamard(4)

# Convert to sequency order
H_seq = sequency_order(H)

# Generate 2D Walsh-Hadamard basis
def walsh_hadamard_basis(H):
    n = H.shape[0]
    basis = []
    for u in range(n):
        for v in range(n):
            basis.append(np.outer(H[u], H[v]))
    return basis

basis = walsh_hadamard_basis(H_seq)

# Plot 16 basis functions (4x4 grid)
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(basis[i], cmap='gray', vmin=-1, vmax=1)
    ax.axis('off')

    # Add border (rectangle)
    rect = patches.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        linewidth=1.5, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

plt.tight_layout()
plt.show()


''' PROBLEM 4:  8X8 DCT BASIS '''

N = 8
basis_functions = dct_basis(N)

# Plot 64 basis functions (8x8 grid)
fig, axes = plt.subplots(N, N, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(basis_functions[i], cmap='gray')
    ax.axis('off')

    # Add border (rectangle)
    rect = patches.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        linewidth=1.5, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

plt.tight_layout()
plt.show()


''' PROBLEM 5:  AREA OF CHAIN CLOSURE '''

# 8-connectivity direction vectors
directions = {
    0: (1, 0),   1: (1, -1),  2: (0, -1),  3: (-1, -1),
    4: (-1, 0),  5: (-1, 1),  6: (0, 1),   7: (1, 1)
}

def chain_code_area(chain_code):
    # Convert string to list of integers
    chain = [int(c) for c in str(chain_code)]
    
    # Starting point
    x, y = 0, 0
    points = [(x, y)]
    
    # Build boundary points
    for d in chain:
        dx, dy = directions[d]
        x += dx
        y += dy
        points.append((x, y))
    
    # Shoelace formula
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    
    return int(area)

print(chain_code_area(217644))
