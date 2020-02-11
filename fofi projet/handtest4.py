# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:25:26 2019



@author: theo*
"""

import cv2
import numpy as np
import math

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


while True:
    _, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    
    lower_skin = np.array([l_h, l_s, l_v])
    upper_skin = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    resultfilter = cv2.medianBlur(mask, 5)
    result = cv2.bitwise_and(frame, frame, mask=resultfilter)

    cv2.imshow('result',result)
    
    im2, contours, hierarchy = cv2.findContours(resultfilter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
    cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    cv2.imshow("image", frame)
    
    if len(contours) > 0:
        cnt=contours[0]
        hull = cv2.convexHull(cnt, returnPoints=False)
		# finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
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
            
        cv2.imshow('img',result)
        cv2.imshow('img1',frame)
        #cv2.imshow('img2',img2)

        
    
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()   