import numpy as np
import cv2
from matplotlib import pyplot as plt


pierre = cv2.imread('cailasse.jpg')
img1 = cv2.resize(pierre,(480,640))
papier = cv2.imread('feuille.jpg')
img2 = cv2.resize(papier,(480,640))
ciseau = cv2.imread('coupecoupe.jpg')
img3 = cv2.resize(ciseau,(480,640))
latasse = cv2.imread('latasse.jpg')
img4 = cv2.resize(latasse,(480,640))

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(img3,None)
kp4, des4 = sift.detectAndCompute(img4,None)


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

#register = cv2.VideoCapture(0)
#while True:
#    ret, frame = register.read()
#    if (ret == False):
#        print("Error reading from camera")
#        break
#    cv2.imshow('frame',frame)
#    key = cv2.waitKey(1) & 0xFF;
#    if (key == ord('q') or key == ord('x')):
#        break;
#register.release()
#cv2.destroyAllWindows()


