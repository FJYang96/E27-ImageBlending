##############################################
# Project 2 - Laplacian blending
# Fengjun Yang(Jack), Weite Liu
##############################################

import cv2
import numpy as np
from copy import deepcopy

def pyr_build(orig, depth):
    """ This function takes in an image and the desired depth to create
    a laplacian pyramid represented in the form of a list"""
    img = deepcopy(orig)
    G = [img]       # G is the list representing the gaussian pyramid
    lp = []          # lp is the list representing the laplacian pyramid

    for i in range(depth-1):
        # First get the next gaussian
        next_G = cv2.pyrDown(G[i])
        G.append(next_G)
        # Then find the laplacian
        resized_G = cv2.pyrUp(G[i+1], dstsize=(G[i].shape[1], G[i].shape[0]) )
        float_G = resized_G.astype(np.float32)
        l = G[i] - float_G
        lp.append(l)

    lp.append( G[depth-1] )

    return lp

def pyr_reconstruct(lp_orig):
    """This function takes in a list representing the laplacian pyramid
    of an image and then return the image reconstructed from this pyramid"""
    lp = deepcopy(lp_orig)     # First make a deepcopy of the laplacian pyr
    R = lp.pop()              # The first R is the last element in lp

    while lp:
        shape = lp[-1].shape[1],lp[-1].shape[0]
        R = cv2.pyrUp(R,dstsize=shape)
        R = R.astype(np.float32)
        R += lp.pop()
        '''
        cv2.imshow('window',0.5 + 0.5*(R / np.abs(R).max() ))
        while cv2.waitKey() < 0 : pass
        '''

    # Convert the image into integer format
    R = np.clip(R, 0, 255)
    R = np.uint8(R) 

    return R

def alpha_blend(A, B, alpha_orig):
    """ Performs an alpha blend on two images A and B with mask alpha
    returns the blended image"""
    alpha = deepcopy(alpha_orig)

    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)

    '''
    cv2.imshow('window',0.5 + 0.5*(A / np.abs(A).max() ))
    while cv2.waitKey() < 0 : pass
    cv2.imshow('window',0.5 + 0.5*(B / np.abs(B).max() ))
    while cv2.waitKey() < 0 : pass
    '''

    # If RGB, expend alpha to be also three dimensional
    if len(A.shape) == 3:
        alpha = np.expand_dims(alpha, 2)

    return A + alpha * (B - A)

def lap_blend(lpA_orig, lpB_orig, alpha):
    """Performs an alpha blend w/ the lapacian pyramid of two images
    returns resulted blended image"""
    lpA = deepcopy(lpA_orig)
    lpB = deepcopy(lpB_orig)
    new_lp = []
    # Blend each level of the images
    while lpA:
        A = lpA.pop(0)
        B = lpB.pop(0)
        # resize alpha mask to be of the same shape as A and B
        shape = A.shape[1], A.shape[0]
        temp_alpha = cv2.resize(alpha, shape)
        new_lp.append( alpha_blend(A, B, temp_alpha) )

    # build the blended image from the new laplacian
    return pyr_reconstruct(new_lp)

# Here starts the main program

# Get the images
img1 = cv2.imread('wendy.jpg')
#img1 = 0.5 + 0.5 * (img1 / np.abs(img1.max()) )
img2 = cv2.imread('burger.jpg')
shape = img1.shape[0], img1.shape[1] # Assume 2 images have same shape

# Test for building a pyramid
lp_depth = 4    # The depth of the laplacian pyramid
lp1 = pyr_build(img1, lp_depth)
"""
# This would show the pyramid
for L in lp1:
    cv2.imshow('window',0.5 + 0.5*(L / np.abs(L).max() ))
    while cv2.waitKey() < 0 : pass
"""
lp2 = pyr_build(img2, lp_depth)

"""
# Test reconstructing an image from its laplacian
R1 = pyr_reconstruct(lp1)
cv2.imshow('window',R1)
while cv2.waitKey() < 0 : pass
"""

# Creating a mask
alpha = np.zeros(shape, np.float32)
cv2.ellipse(alpha, (512, 384), (150,200), 0, 0, 360, 1, -1)
alpha = cv2.GaussianBlur(alpha, (0,0), 5)

# Blending images directly through a alpha blend
direct_blend = alpha_blend(img1, img2, alpha)
img1 = img1.astype(np.float32)
cv2.imshow('window',0.5 + 0.5*(img1) / 255 )
while cv2.waitKey() < 0 : pass

cv2.imshow('window',0.5 + 0.5*(direct_blend / np.abs(direct_blend).max() ))
while cv2.waitKey() < 0 : pass

# Blending images using laplacian
l_blend = lap_blend(lp1, lp2, alpha)
cv2.imshow('window', l_blend)
while cv2.waitKey() < 0 : pass
