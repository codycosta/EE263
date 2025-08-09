'''
Author: Cody Costa
Date:   8/5/2025

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def main(gaussian_filter: tuple, c: int, block_size: int):
    # load image into grayscale form
    img = 'defects.png'
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # reduce noise with gaussian blur
    blurred = cv2.GaussianBlur(gray_enhanced, gaussian_filter, 0)


    ''' LIGHT AREA BUMP DEFECTS '''
    # playing around with these values to tune results
    C = -4
    BLOCK_SIZE = 21

    # binarize image
    # bump_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C)
    bump_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

    # find contours
    bump_contours, _ = cv2.findContours(bump_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)


    ''' DARK AREA CRATER DEFECTS '''
    # playing around with these values to tune results
    C = 16
    BLOCK_SIZE = 25

    # binarize image
    crater_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, BLOCK_SIZE, C)

    # find contours
    crater_contours, _ = cv2.findContours(crater_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)


    # record centroids of each defect
    centroids = []

    contours = bump_contours + crater_contours


    # display image processing steps
    # cv2.imshow('original', image)
    # cv2.imshow('Gray + enhanced contrast', gray_enhanced)
    # cv2.imshow('gaussian smoothed', blurred)
    # cv2.imshow('light area binary', bump_thresh)
    # cv2.imshow('dark area binary', crater_thresh)


    # plot defects on original image
    for contour in contours:
        M = cv2.moments(contour)

        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            centroids.append((cx, cy))

            # cv2.circle(blurred, (cx, cy), 5, (0, 255, 0), 2)
            
        cv2.drawContours(blurred, [contour], -1, (0, 0, 255), 1)


        
    # write results to blurred photo and display mapped defects
    # cv2.imwrite('defects_labeled_improved.jpg', blurred)
    cv2.imshow(f'{gaussian_filter=}, {c=}, {block_size=}', blurred)
    # print defect count
    print(f'defect count = {len(contours)}')



    # close windows with keypress
    cv2.waitKey()
    cv2.destroyAllWindows()


Gaussian_Filters = [(5, 5), (7, 7)]
BLOCKS = [19, 21, 23]
C_vals = [-3, -4, -5]

for F in Gaussian_Filters:
    for B in BLOCKS:
        for C in C_vals:
            main(F, C, B)