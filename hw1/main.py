'''
Author: Cody Costa
Date:   7/14/2025

'''

import cv2


''' PROBLEM 1 '''

# filename to open
LENA = 'lena.webp'

# read image to memory
img = cv2.imread(LENA)

# display image
cv2.imshow('image', img)

# await key press to close program
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# display gray image
cv2.imshow('gray image', img_gray)

# await key press to close program
cv2.waitKey(0)
cv2.destroyAllWindows()


''' PROBLEM 2 '''
