'''
Author: Cody Costa
Date:   7/14/2025

'''

import cv2
import matplotlib.pyplot as plt
import numpy as np


''' PROBLEM 1:  WARMUP, DISPLAY GS AND COLOR IMG '''
# filename to open
LENA = 'lena.webp'

# read image to memory
img = cv2.imread(LENA)

# display image
cv2.imshow('image', img)

# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# display gray image
cv2.imshow('gray image', img_gray)

# await key press to close program
cv2.waitKey(0)
cv2.destroyAllWindows()



''' PROBLEM 2:  SOBEL AND PREWITT EDGE DETECTION '''
# Sobel edge detection using OpenCV built in functions

# apply 3x3 x and y Sobel kernels
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

# calculate magnitude and normalize results into an 8 bit image
sobel_mag = np.hypot(sobel_x, sobel_y)
sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



# Prewitt edge detection using manual kernel and filter2D

# apply x and y Prewitt kernels 
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
# prewitt_y = np.array([[1, 1, 1],
#                       [0, 0, 0],
#                       [-1, -1, -1]], dtype=np.float32)
prewitt_y = prewitt_x.T

# apply convolution
px = cv2.filter2D(img_gray.astype(np.float32), -1, prewitt_x)
py = cv2.filter2D(img_gray.astype(np.float32), -1, prewitt_y)

# calculate magnitude and normalize results into an 8 bit image
prewitt_mag = np.hypot(px, py)
prewitt_mag = cv2.normalize(prewitt_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



# display results
combined = np.hstack((sobel_mag, prewitt_mag))

plt.figure(figsize=(10, 5))
plt.imshow(combined, cmap='gray')
plt.title('Sobel (left) vs Prewitt (right)')
plt.axis('off')
plt.show()


import enhancement_pipline

''' PROBLEM 3:  IMAGE ENHANCEMENT PIPELINE '''
# new photo to load
BONES = 'Fig3-46a.tif'

# accessory file created to handle enhancement logic pipeline
enhancement_pipline.enhance_img(BONES)


''' PROBLEM 4:  IMAGE ENHANCEMENT PIPELINE '''
# new photo to load
MRI = 'Image9.jpg'

# accessory file created to handle enhancement logic pipeline
enhancement_pipline.enhance_img(MRI)
