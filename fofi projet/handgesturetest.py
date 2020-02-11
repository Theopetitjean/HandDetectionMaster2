# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:03:41 2019

@author: theo*
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('feuille.jpg',0)
img1 = cv2.resize(img1,(480,640))
hpinc = cv2.imread('hp_insc.jpg',0)

img2 = cv2.resize(hpinc,(480,640))
#img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
ret,img1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY_INV)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

#img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)        
# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(400)    
# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img1,None)    
imga = cv2.drawKeypoints(img1,kp,None)   
cv2.imshow('surf feature',imga)


#img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)        
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()        
# find and draw the keypoints
kp = fast.detect(img1,None)
imgb = cv2.drawKeypoints(img1, kp, None, color=(255,0,0))        
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv2.imshow('fast_true.png',imgb)           
 

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

plt.imshow(img3),plt.show()
cv2.imshow('uneimageincroyable',img3)

cv2.waitKey(0)
cv2.destroyAllWindows()