'''
Author: Cody Costa
Date:   8/5/2025

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


# load image into grayscale form
img = 'defects.png'
image = cv2.imread(img)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_enhanced = clahe.apply(gray)

# reduce noise with gaussian blur
blurred = cv2.GaussianBlur(gray_enhanced, (7, 7), 0)


# playing around with these values to tune results
C = 16
BLOCK_SIZE = 25

# binarize image
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, BLOCK_SIZE, C)

# find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)

# record centroids of each defect
crater_centroids = []

# plot defects on original image
for contour in contours:
    M = cv2.moments(contour)

    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        crater_centroids.append((cx, cy))

        cv2.circle(blurred, (cx, cy), 5, (0, 255, 0), 1)
        
    # cv2.drawContours(blurred, [contour], -1, (0, 0, 255), 3)

# cv2.imwrite('defects_labeled_improved.jpg', blurred)

# cv2.imshow('GE', gray_enhanced)
# cv2.imshow('blurred', blurred)
# cv2.imshow('binary', thresh)
# cv2.imshow('defects', image)

# print defect count
# print(f'defect count = {len(crater_centroids)}')



# close windows with keypress
cv2.waitKey()
cv2.destroyAllWindows()
