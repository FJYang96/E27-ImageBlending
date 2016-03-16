##############################################
# Project 2 - Laplacian blending
# Fengjun Yang(Jack), Weite Liu
##############################################

import cv2
import numpy as np
from copy import deepcopy

def pyr_build(orig, depth, show=False):
    """ This function takes in an image and the desired depth to create
    a laplacian pyramid represented in the form of a list"""
    img = deepcopy(orig)
    G = [img]       # G is the list representing the gaussian pyramid
    lp = []          # lp is the list representing the laplacian pyramid

    for i in range(depth-1):
        # First get the next gaussian
        next_G = cv2.pyrDown(G[i])   #dtype is uint8       
        G.append(next_G)
        # Then find the laplacian
        resized_G = cv2.pyrUp(G[i+1], dstsize=(G[i].shape[1], G[i].shape[0]) )
        float_G = resized_G.astype(np.float32)
        l = G[i].astype(np.float32) - float_G
        lp.append(l)
        if show:
            cv2.imshow('Building Pyramid',0.5 + 0.5*(l / np.abs(l).max() ))
            while cv2.waitKey() < 0 : pass

    lp.append( G[depth-1].astype(np.float32) )

    return lp

def pyr_reconstruct(lp_orig, show=False):
    """This function takes in a list representing the laplacian pyramid
    of an image and then return the image reconstructed from this pyramid"""
    lp = deepcopy(lp_orig)     # First make a deepcopy of the laplacian pyr
    R = lp.pop()              # The first R is the last element in lp

    while lp:
        shape = lp[-1].shape[1],lp[-1].shape[0]
        R = cv2.pyrUp(R,dstsize=shape)
        R = R.astype(np.float32)
        R += lp.pop()
        if show:
            cv2.imshow('Reconstructing',0.5 + 0.5*(R / np.abs(R).max() ))
            while cv2.waitKey() < 0 : pass

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

def lap_blend(lpA_orig, lpB_orig, alpha, show=False):
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
        temp_alpha = cv2.resize(alpha, shape, interpolation=cv2.INTER_AREA)
        new_lp.append( alpha_blend(A, B, temp_alpha) )

    # build the blended image from the new laplacian
    return pyr_reconstruct(new_lp, show)

##### Main program #####
# Ask user if she wants to see the intermediate results
print "Hello! This program blends Hilary Clinton and Bernie Sanders... faces"
print "If you want to see intermediate steps, please enter 'y'."
ifShow = False
answer = raw_input()
if answer == 'y':
    ifShow = True

# Get the images
print "Reading in images"
img1 = cv2.imread('clinton.jpg')
img2 = cv2.imread('bernie.jpg')
print "Successfully read in images"

# Assume 2 images have same shape
shape = img1.shape[0], img1.shape[1]

# Test for building a pyramid
lp_depth = 8   # The depth of the laplacian pyramid
print "Building a pyramid for the first image"
lp1 = pyr_build(img1, lp_depth, show=ifShow)
print "Building a pyramid for the second image"
lp2 = pyr_build(img2, lp_depth, show=ifShow)

# Show reconstructing an image from its laplacian
if ifShow:
    print "Example for reconstructing from pyramid"
    R1 = pyr_reconstruct(lp1, show=True)
    cv2.imshow('Reconstructed',R1)
    while cv2.waitKey() < 0 : pass

# Creating a mask
alpha = np.zeros(shape, np.float32)
"""
# 1. Mask for wendy and burger
cv2.ellipse(alpha, (512, 384), (150,200), 0, 0, 360, 1, -1)
"""
# 2. Mask for Bernie and Clinton
cv2.ellipse(alpha, (175, 150), (52, 82), 0, 0, 360, 1, -1)

# Show the mask
if ifShow:
    print "Showing the mask"
    balpha = alpha.astype(np.uint8).view(np.bool)
    masked = np.empty((shape[1],shape[0],3), np.uint8)
    masked[:] = 0
    masked[balpha] = img2[balpha]
    cv2.imshow('Mask',masked)
    while cv2.waitKey() < 0: pass
    cv2.imwrite('mask.jpg',alpha*255)

# alpha = cv2.GaussianBlur(alpha, (0,0), 5)

# Blending images directly through a alpha blend
direct_blend = alpha_blend(img1, img2, alpha)
direct_blend = direct_blend.astype(np.uint8)
print "The result of direct_blend"
cv2.imshow('Direct blending', direct_blend)
while cv2.waitKey() < 0 : pass

# Blending images using laplacian
print "The result of blending using laplacian"
l_blend = lap_blend(lp1, lp2, alpha, show=ifShow)
cv2.imshow('laplacian blending', l_blend)
cv2.imwrite('Hillary_Sanders.jpg',l_blend)
while cv2.waitKey() < 0 : pass

"""
# Build a pyramid for hybrid 
img3 = cv2.imread('hybrid_result.jpg')
lp = pyr_build(img3, 6, show=True)
counter = 0
for l in lp:
    filename = 'pyramid' + str(counter) + '.jpg'
    l = l * 255 / l.max()
    l = np.clip(l, 0, 255)
    l = np.uint8(l) 
    cv2.imwrite(filename,l)
    counter += 1
"""

# Test blending for big chunks of color vs small dots
img4 = cv2.imread('blue.jpg')
img5 = cv2.imread('orange.jpg')
shape = img4.shape[1],img4.shape[0]
alpha = np.empty(shape, 'float32')
alpha = cv2.rectangle(alpha, (0,0), (shape[1]/2,shape[0]), 1, -1)
lp4 = pyr_build(img4, 5)
lp5 = pyr_build(img5, 5)
l_blend = lap_blend(lp4, lp5, alpha)
cv2.imshow('color', l_blend)
cv2.imwrite('solid_color.jpg', l_blend)
while cv2.waitKey() < 0 : pass
