'''
Author: Cody Costa
Date:   7/14/2025

'''


import cv2
import matplotlib.pyplot as plt
import numpy as np

def enhance_img(IMAGE):

    ''' IMAGE A '''
    # new photo to load
    # BONES = 'Fig3-46a.tif'

    # read img to memory
    try:
        img = cv2.imread(IMAGE)
    except Exception:
        print(f'Could not read {IMAGE}')
        return

    # convert to grayscale (even though it kinda already is)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


    # ///////////////////////////////////////////////////////////////////////////////////////////////
    ''' IMAGE B '''
    # compute the laplacian of the image
    laplacian = cv2.Laplacian(img_gray, ddepth=cv2.CV_32F, ksize=3)
    # normalize and format to 8 bit display
    laplacian_8bit = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)


    # ///////////////////////////////////////////////////////////////////////////////////////////////
    ''' IMAGE C '''
    # sharpen image by adding laplacian to base grayscale photo (IMAGE A + IMAGE B)
    alpha = 1
    sharpened = img_gray - (alpha * laplacian_8bit)
    # normalize and format to 8 bit display
    sharpened = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)


    # ///////////////////////////////////////////////////////////////////////////////////////////////
    ''' IMAGE D '''
    # apply 3x3 x and y Sobel kernels to IMAGE A
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # calculate magnitude and normalize results into an 8 bit image
    sobel_mag = np.hypot(sobel_x, sobel_y)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    # ///////////////////////////////////////////////////////////////////////////////////////////////
    ''' IMAGE E '''
    # apply 5x5 averaging filter to Sobel (IMAGE D)
    smoothed = cv2.blur(sobel_mag, (5, 5))


    # ///////////////////////////////////////////////////////////////////////////////////////////////
    ''' IMAGE F '''
    # mask image formed by the product of IMAGE C and IMAGE E
    product = sharpened.astype(np.float32) * smoothed.astype(np.float32)
    product_normalized = cv2.normalize(product, None, 0, 255, cv2.NORM_MINMAX)
    product_normalized = product_normalized.astype(np.uint8)


    # ///////////////////////////////////////////////////////////////////////////////////////////////
    ''' IMAGE G '''
    # sharpened image created by the sum of IMAGE A and IMAGE F
    sharpened_mask = img_gray + (alpha * product_normalized.astype(np.float32))
    # sharpened_mask = cv2.normalize(sharpened_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sharpened_mask_norm = np.clip(sharpened_mask, 0, 255).astype(np.uint8)


    # ///////////////////////////////////////////////////////////////////////////////////////////////
    ''' IMAGE H '''
    # power law transform to IMAGE G
    combined_norm = cv2.normalize(sharpened_mask, None, 0, 1.0, cv2.NORM_MINMAX)
    gamma = 0.5
    gamma_corrected = np.power(combined_norm, gamma)
    # convert to uint8 image
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)


    # display images
    cv2.imshow('I', img_gray.astype(np.uint8))
    cv2.imshow('Laplacian', laplacian_8bit.astype(np.uint8))
    cv2.imshow('Sharpened', sharpened)
    cv2.imshow('Sobel3x3', sobel_mag)
    cv2.imshow('Smoothed Sobel', smoothed)
    cv2.imshow('Mask', product_normalized)
    cv2.imshow('Sharp Mask', sharpened_mask_norm)
    cv2.imshow('Power Transform', gamma_corrected)

    # await key press to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
