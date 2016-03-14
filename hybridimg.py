###################################################
# Project 2 - Hybrid image
# Fengjun Yang(Jack), Weite Liu
###################################################

import cv2
import numpy as np

def lopass(orig, sigma):
    shape = orig.shape
    img = np.empty(shape, np.float32)
    # Apply a lowpass filter on the image
    img = cv2.GaussianBlur(img, (0,0), sigmaX = sigma)
    return img

def hipass(orig, sigma):
    img = lopass(orig, sigma)
    return orig.astype(np.float32) - img

# Read in source images
A = cv2.imread('apple.jpg')
B = cv2.imread('orange.jpg')

# Parameters
Ka = 1
Kb = 1
sigmaA = 5
sigmaB = 5

# Compute the hybrid image
hybrid_img = Ka * lopass(A, sigmaA) + Kb * hipass(B, sigmaB)

# Convert the image into integer
hybrid_img = np.clip(hybrid_img, 0, 255)
hybrid_img = np.uint8(hybrid_img)

# Show the image
cv2.imshow('window', hybrid_img)
while cv2.waitKey() < 0: pass
