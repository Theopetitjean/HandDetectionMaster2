# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:26:50 2019

@author: theo*
"""

import numpy as np 
import cv2 
import math

#lower_skin = np.array([l_h, l_s, l_v])
#upper_skin = np.array([u_h, u_s, u_v])
    
#lower_skin = cv2.Vec(hue, 0.8 * 255, 0.6 * 255);
#upper_skin = cv2.Vec(hue, 0.1 * 255, 0.05 * 255);

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #hlsframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    #cv2.imshow('test',hlsframe)
    
    blur = cv2.GaussianBlur(frame, (3,3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82])
    upper_color = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blur = cv2.medianBlur(mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    hsv_d = cv2.dilate(blur, kernel)
    
    cv2.imshow('lolill',hsv_d)
    im2, contours, hierarchy = cv2.findContours(hsv_d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
    cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    cv2.imshow("image", frame)
    
    #convex hull points
    hull = []
    for i in range (len(contours)):
        #create hull obje for each contour
        hull.append(cv2.convexHull(contours[i],False))
    
     #creat empty img 
    drawing = np.zeros((hsv_d.shape[0],hsv_d.shape[1],3),np.uint8)
    #♪draw contour and hull
    for i in range (len(contours)):
         color = (255,0,0)
         cv2.drawContours(drawing,hull,i,color,1,8)
    cv2.imshow('hull',drawing)
    for i in range (len(contours)):
         color = (255,0,0)
         cv2.drawContours(frame,hull,i,color,1,8)  
    cv2.imshow('hullf', frame)
    
    if len(contours) > 0:
        cnt=contours[0]
        hull2 = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull2)
        count_defects = 0
    		# applying Cosine Rule to find angle for all defects (between fingers)
    		# with angle > 90 degrees and ignore defect
        if defects is not None:
            for i in range(defects.shape[0]):
                p,q,r,s = defects[i,0]
                finger1 = tuple(cnt[p][0])
                finger2 = tuple(cnt[q][0])
                dip = tuple(cnt[r][0])
                # find length of all sides of triangle
                a = math.sqrt((finger2[0] - finger1[0])**2 + (finger2[1] - finger1[1])**2)
                b = math.sqrt((dip[0] - finger1[0])**2 + (dip[1] - finger1[1])**2)
                c = math.sqrt((finger2[0] - dip[0])**2 + (finger2[1] - dip[1])**2)
                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57.29
                # ignore angles > 90 and highlight rest with red dots
                if angle <= 90:
                    count_defects += 1
                    
        #		 define actions required
        if count_defects == 1:
            cv2.putText(frame,"THIS IS 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 2:
            cv2.putText(frame, "THIS IS 3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 3:
            cv2.putText(frame,"This is 4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(frame,"THIS IS 5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                   
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()    