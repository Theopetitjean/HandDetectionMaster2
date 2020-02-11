# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:11:44 2019

test of existing code for detection of the qrcode detecion and reading

roboust method of q code reading, can read 2 and may be more in the same time,

detection is efficient but the distance btw the camera and the code couldbe a prblm 
*

@author: theo*
"""
from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import time

# get the webcam:  
cap = cv2.VideoCapture(0)
time.sleep(2)

def decode(im) : 
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)
    # Print results
    for obj in decodedObjects:
        print('Type : ', obj.type)
        print('Data : ', obj.data,'\n')     
    return decodedObjects

font = cv2.FONT_HERSHEY_SIMPLEX

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         
    decodedObjects = decode(im)

    for decodedObject in decodedObjects: 
        points = decodedObject.polygon
     
        # If the points do not form a quad, find convex hull
        if len(points) > 4 : 
          hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
          hull = list(map(tuple, np.squeeze(hull)))
        else : 
          hull = points;         
        # Number of points in the convex hull
        n = len(hull)    
        
        # Draw the convext hull
        for j in range(0,n):
          cv2.line(frame, hull[j], hull[ (j+1) % n], (255,0,0), 3)

        x = decodedObject.rect.left
        y = decodedObject.rect.top

        print(x, y)

        print('Type : ', decodedObject.type)
        print('Data : ', decodedObject.data,'\n')

        barCode = str(decodedObject.data)
        cv2.putText(frame, barCode, (x, y), font, 1, (0,255,255), 2, cv2.LINE_AA)
               
    # Display the resulting frame
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break   

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()