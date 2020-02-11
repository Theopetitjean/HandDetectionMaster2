# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:15:58 2019

@author: Utilisateur
"""

import cv2
import numpy as np

def nothing(x):
    pass


lafeuille = cv2.imread("feuille.jpg")
lafeuille = cv2.cvtColor(lafeuille, cv2.COLOR_BGR2GRAY)
ret,lafeuillebin = cv2.threshold(lafeuille,130,255,cv2.THRESH_BINARY_INV)
lafeuillebin = cv2.resize(lafeuillebin,(480,640))
lafeuilleres = cv2.medianBlur(lafeuillebin, 5)
cv2.imshow('testmi',lafeuillebin)

lacailasse = cv2.imread("cailasse.jpg")
lacailasse = cv2.cvtColor(lacailasse, cv2.COLOR_BGR2GRAY)
ret,lacailassebin = cv2.threshold(lacailasse,130,255,cv2.THRESH_BINARY_INV)
lacailassebin = cv2.resize(lacailassebin,(480,640))
cailasseres = cv2.medianBlur(lacailassebin, 5)
cv2.imshow('testfu',cailasseres)

lesixeau = cv2.imread("coupecoupe.jpg")
lesixeau = cv2.cvtColor(lesixeau, cv2.COLOR_BGR2GRAY)
ret,lesixeaubin = cv2.threshold(lesixeau,130,255,cv2.THRESH_BINARY_INV)
lesixeaubin = cv2.resize(lesixeaubin,(480,640))
lesixeaures = cv2.medianBlur(lesixeaubin, 5)
cv2.imshow('testci',lesixeaures)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
#cv2.namedWindow("roi decision")
#cv2.createTrackbar("ROI selector y or no ","Trackbars", 0, 1, nothing)

while True:
    _, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    #tuveuxoutuveuxpas = cv2.getTrackbarPos("ROI sel","trackbars")
    
    lower_skin = np.array([l_h, l_s, l_v])
    upper_skin = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    resultfilter = cv2.medianBlur(mask, 5)
    result = cv2.bitwise_and(frame, frame, mask=resultfilter)

    
    #cv2.imshow("frame", frame)
    #cv2.imshow('filtered mask ?', resultfilter)
    cv2.imshow('result',result)
    
    im2, contours, hierarchy = cv2.findContours(resultfilter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
    cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    cv2.imshow("image", frame)
    
    #convex hull points
    hull = []
    for i in range (len(contours)):
        #create hull obje for each contour
        hull.append(cv2.convexHull(contours[i],False))
    
    #creat empty img 
    drawing = np.zeros((result.shape[0],result.shape[1],3),np.uint8)
    #♪draw contour and hull
    for i in range (len(contours)):
         color = (255,0,0)
         cv2.drawContours(frame,hull,i,color,1,8)  
    cv2.imshow('hullf', frame)


    
    #if tuveuxoutuveuxpas == 1:
    #    ROI = cv2.selectROI(stormtrooper)
    #    cv2.imshow("roi",ROI)
    #    if ROI != (0,0,0,0):
    #        cropcrop = stormtrooper[int(ROI[1]):int(ROI[1]+ROI[3]),int(ROI[0]):int(ROI[0]+ROI[2])]
    #        cv2.imshow('jaimelesbanane',cropcrop)
    
        
    
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()    