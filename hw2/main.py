'''
Author: Cody Costa
Date:   7/31/2025

'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided


# ''' PROBLEM 1:  FILTER COMPARISONS '''

LENA = 'lena.webp'
img = cv2.imread(LENA)

# salt and pepper noise function
def add_salt_pepper(image):
    noisy_img = image.copy()
    total_defects = 150

    for d in range(total_defects):
        coordinate = (np.random.randint(0, noisy_img.shape[0]), np.random.randint(0, noisy_img.shape[1]))
        defect = np.random.randint(0, 2)

        if defect == 0:
            noisy_img[coordinate[0], coordinate[1]] = [0, 0, 0]

        else:
            noisy_img[coordinate[0], coordinate[1]] = [255, 255, 255]

    return noisy_img

# gaussian noise function
def add_gaussian_noise(image, mean, sigma):
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), gaussian)

    return np.clip(noisy, 0, 255).astype(np.uint8)


salty = add_salt_pepper(img)
gaussian_and_salty = add_gaussian_noise(salty, 0, 20)

cv2.imshow('original', img)
cv2.imshow('salty', salty)
cv2.imshow('gaussian', gaussian_and_salty)

# test median filter
median_only = cv2.medianBlur(gaussian_and_salty, 5)

# test smoothing filter
smoothing_only = cv2.blur(gaussian_and_salty, (5, 5))

# combined, median-smoothing filter
combo = cv2.blur(cv2.medianBlur(gaussian_and_salty, 5), (5, 5))


# display images
cv2.imshow('gaussian', gaussian_and_salty)
cv2.imshow('median', median_only)
cv2.imshow('smoothed', smoothing_only)
cv2.imshow('combined', combo)


# # close windows when done
cv2.waitKey(0)
cv2.destroyAllWindows()

# IN THIS CASE, THE MEDIAN FILTER GIVES THE SHARPEST IMAGE OF THE 3 FILTERS


''' PROBLEM 2:  RANDOM NOISE '''

def add_random_noise(image, noise_level):
    noise = np.random.randint(-noise_level, noise_level, image.shape, dtype='int16')

    noisy_img = image.astype('int16') + noise
    return np.clip(noisy_img, 0, 255).astype('uint8')

# add noise to LENA photo
noised_up_img = add_random_noise(img, 60)
cv2.imshow('noise', noised_up_img)

# apply gaussian filter
smoothed_img = cv2.GaussianBlur(noised_up_img, (5, 5), 0)
cv2.imshow('smooth', smoothed_img)

cv2.waitKey(0)
cv2.destroyAllWindows()



''' PROBLEM 3:  BILATERAL/NAGAO FILTERS '''

BONES = 'Fig3-46a.tif'
img2 = cv2.imread(BONES)

# apply bilateral filter
bilateral = cv2.bilateralFilter(img2, 9, 75, 75)

# apply nagao filter
def nagao_matsuyama_filter(image, window_size=5):
    pad = window_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    output = np.zeros_like(image)

    # Define 9 regions in the 5x5 window
    regions = [
        [(0, 0), (2, 2)],  # top-left
        [(0, 1), (2, 3)],  # top-center
        [(0, 2), (2, 4)],  # top-right
        [(1, 0), (3, 2)],  # mid-left
        [(1, 1), (3, 3)],  # center
        [(1, 2), (3, 4)],  # mid-right
        [(2, 0), (4, 2)],  # bottom-left
        [(2, 1), (4, 3)],  # bottom-center
        [(2, 2), (4, 4)]   # bottom-right
    ]

    for y in range(pad, padded.shape[0] - pad):
        for x in range(pad, padded.shape[1] - pad):
            window = padded[y - pad:y + pad + 1, x - pad:x + pad + 1]

            best_region_mean = np.zeros(3)
            min_variance = np.inf

            for (y0, x0), (y1, x1) in regions:
                region = window[y0:y1 + 1, x0:x1 + 1]
                variance = np.var(region)
                if variance < min_variance:
                    min_variance = variance
                    best_region_mean = np.mean(region, axis=(0, 1))

            output[y - pad, x - pad] = best_region_mean

    return output.astype(np.uint8)

nagao = nagao_matsuyama_filter(img2)

cv2.imshow('bones', img2)
cv2.imshow('bilateral', bilateral)
cv2.imshow('nagao', nagao)

cv2.waitKey(0)
cv2.destroyAllWindows()


''' PROBLEM 4:  BILATERAL FILTER TWEAKS '''

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)

def vectorized_bilateral_div(image, diameter=5, sigma_space=12, method='inv'):
    pad = diameter // 2
    img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = image.shape

    # Create sliding windows using strides
    shape = (h, w, diameter, diameter)
    strides = (img.strides[0], img.strides[1], img.strides[0], img.strides[1])
    windows = as_strided(img, shape=shape, strides=strides)

    # Precompute Gaussian for distance (G(D))
    ax = np.arange(-pad, pad + 1)
    xx, yy = np.meshgrid(ax, ax)
    G_D = np.exp(-(xx**2 + yy**2) / (2 * sigma_space**2)).astype(np.float32)  # (k, k)
    G_D = G_D[None, None, :, :]  # (1, 1, k, k)

    # Center pixel of each window
    center = img[pad:pad+h, pad:pad+w].astype(np.float32)[:, :, None, None]

    # Intensity difference
    diff = np.abs(windows.astype(np.float32) - center) + 1e-5

    # Compute weights using division instead of Gaussian(I)
    if method == 'inv':
        weights = G_D / diff
    elif method == 'inv2':
        weights = G_D / (diff**2)
    else:
        raise ValueError("method must be 'inv' or 'inv2'")

    # Normalize weights
    weights /= np.sum(weights, axis=(2, 3), keepdims=True)

    # Weighted sum for filtering
    result = np.sum(weights * windows, axis=(2, 3))

    return np.clip(result, 0, 255).astype(np.uint8)

# Apply OpenCV bilateral filter for reference
bilateral_cv2 = cv2.bilateralFilter(gray.astype(np.uint8), 9, 75, 75)

# Apply custom division-based filters
bilateral_inv = vectorized_bilateral_div(gray, diameter=5, sigma_space=12, method="inv")
bilateral_inv2 = vectorized_bilateral_div(gray, diameter=5, sigma_space=12, method="inv2")

cv2.imshow('bilateral cv2', bilateral_cv2)
cv2.imshow('inv', bilateral_inv)
cv2.imshow('inv2', bilateral_inv2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# IN THIS CASE, THE CUSTOM VECTORIZED BILATERAL FILTERS PRESERVE EDGES BETTER THAN THE REGULAR FILTER DOES


''' PROBLEM 5:  ENTROPY '''

# 8x8 image matrix
G = np.array([
    [139, 144, 149, 153, 155, 155, 155, 155],
    [144, 151, 153, 156, 159, 156, 156, 156],
    [150, 155, 160, 163, 158, 156, 156, 156],
    [159, 161, 162, 160, 160, 159, 159, 159],
    [159, 160, 161, 162, 162, 155, 155, 155],
    [161, 161, 161, 161, 160, 157, 157, 157],
    [162, 162, 161, 163, 162, 157, 157, 157],
    [162, 162, 161, 161, 163, 158, 158, 158]
])

# RMSE approximation (Gaussian noise sigma=5)
sigma = 5
rmse = sigma
psnr = 20 * np.log10(255 / rmse)

# Entropy
unique, counts = np.unique(G, return_counts=True)
probs = counts / G.size
entropy = -np.sum(probs * np.log2(probs))

print("RMSE:", rmse)            # 5
print("PSNR (dB):", psnr)       # 34.15 or 10pi
print("Entropy:", entropy)      # 3.57
