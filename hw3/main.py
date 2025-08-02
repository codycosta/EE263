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

def dct_basis(n=4):
    # init empty 2D array
    basis = np.zeros([n, n, n, n])

    # compute 2D idct for each element in array
    for u in range(n):
        for v in range(n):
            coefficient = np.zeros([n, n])
            coefficient[u, v] = 1
            basis[u, v] = idctn(coefficient, type=2, norm='ortho')

    return basis

N = 4
basis_funcs = dct_basis(N)

# Plot 16 basis functions
fig, axes = plt.subplots(N, N, figsize=(6, 6))
for u in range(N):
    for v in range(N):
        ax = axes[u, v]
        ax.imshow(basis_funcs[u, v], cmap='gray', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"({u},{v})", fontsize=8)

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
