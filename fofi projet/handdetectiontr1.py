# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 08:43:51 2019

hand detection try 1 

@author: theo*
"""

import numpy as np 
import cv2 


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    input_img = frame
    blurred_img = cv2.GaussianBlur(input_img,(5,5),0)
    grayframe = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    ret,imgbin = cv2.threshold(grayframe,127,255,cv2.THRESH_BINARY_INV) # with using the binari img , we will be able to find the countour or ouf 
    # and to draw them, but ht eprobem is that , we need to be in an environment with a smooth color in the background, i'll try to normalyse my img
    # in order to help doing the background substraction ( can a mask be done for that ? )
    cv2.imshow("frame", imgbin)
    blur = cv2.medianBlur(imgbin, 5)
    
    #imgray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(imgbin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
    cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    cv2.imshow("image", frame)
    
    
    #convex hull points
    hull = []
    for i in range (len(contours)):
        #create hull obje for each contour
        hull.append(cv2.convexHull(contours[i],False))
    
     #creat empty img 
    drawing = np.zeros((imgbin.shape[0],imgbin.shape[1],3),np.uint8)
    #♪draw contour and hull
    for i in range (len(contours)):
         color = (255,0,0)
         cv2.drawContours(drawing,hull,i,color,1,8)
    cv2.imshow('hull',drawing)
    for i in range (len(contours)):
         color = (255,0,0)
         cv2.drawContours(frame,hull,i,color,1,8)  
    cv2.imshow('hullf', frame)
    
    #cnts = cv2.findContours(imgbin.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    #_, contours, _ = cv2.findContours(imgbin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   
    #for contour in contours:
     #   cv2.drawContours(frame, contour, -1, (0, 255, 0), 2) 
    #cv2.imshow('lulz', imgbin)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()