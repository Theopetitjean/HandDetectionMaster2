"""
Created on Fri Apr 19 17:29:36 2019

@author: theo*

try to detect and track an object using camshift 
"""

import cv2 
import numpy as np 
from matplotlib import pyplot as plt

imageref = cv2.imread("camsift_tr2.jpg")   # image a changer ne fonctionne pas avec cette methode 
roi = imageref[186: 415, 373: 493]
cv2.imshow("jemangedespates",roi)
x = 373
y = 186
width = 415 - x
height = 493 - y


hsv_histo = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # for some reason hsv convertion do not work here ?  why ?  when gray work well
cv2.imshow("test2",hsv_histo)
histo = cv2.calcHist([hsv_histo], [0], None, [180], [0 , 180])


plt.plot(histo)
plt.xlim([0,256])
plt.show()

 
cap = cv2.VideoCapture(0)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    _, frame = cap.read()
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], histo, [0,180], 1)
    
    ret, track_window = cv2.CamShift(mask, (x, y, width, height), term_criteria)
    
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
    
    cv2.imshow("mask", mask)
    cv2.imshow("frame",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()