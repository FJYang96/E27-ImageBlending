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
    img = cv2.GaussianBlur(orig, (0,0), sigmaX = sigma)
    return img

def hipass(orig, sigma):
    img = lopass(orig, sigma)
    return orig.astype(np.float32) - img

# Read in source images
A = cv2.imread('einstein.jpg', 0)
B = cv2.imread('jobs.jpg', 0)

# Parameters
Ka = 0.9 
Kb = 0.4
sigmaA = 4
sigmaB = 3

# Compute the hybrid image
loA = lopass(A, sigmaA)
loA = Ka * loA
cv2.imshow('window', loA / np.abs(loA.max()) )
while cv2.waitKey() < 0: pass

hiB = hipass(B, sigmaB)
hiB = Kb * hiB
cv2.imshow('window', hiB / np.abs(hiB.max()) )
while cv2.waitKey() < 0: pass

hybrid_img = loA + hiB

# Convert the image into integer
hybrid_img = np.clip(hybrid_img, 0, 255)
hybrid_img = np.uint8(hybrid_img)

# Show the image
cv2.imshow('window', hybrid_img)
while cv2.waitKey() < 0: pass

# Save the image
cv2.imwrite('hybrid_result.jpg', hybrid_img)
