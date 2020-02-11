import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

latasse = cv2.imread('latasse.jpg')
latasse2 = cv2.resize(latasse,(480,640))

ctoutgris = cv2.cvtColor(latasse2,cv2.COLOR_BGR2RGB)

ret,thresh1 = cv2.threshold(ctoutgris,127,255,cv2.THRESH_BINARY_INV)
 
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(thresh1,None)

latasseavecdesfeaturesincroyable = cv2.drawKeypoints(thresh1,kp,None,color=(0,255,0), flags=0)
cv2.imshow('uneimageincroyable',latasseavecdesfeaturesincroyable)

register = cv2.VideoCapture(0)
while True:
    ret, frame = register.read()
    
    if (ret == False):
        print("Error reading from camera")
        break
    
    cv2.imshow('frame',frame)
    #gris = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create()
    kpcam = sift.detect(frame,None)
    #amazfeat = cv2.drawKeypoints(gris,kpcam,None,color=(0,255,0), flags=0)
    # Initiate SIFT detector
    #sift = cv2.SIFT()
    
    
    #FLANN_INDEX_KDTREE = 0
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks = 50)
    
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    #matches = flann.knnMatch(latasseavecdesfeaturesincroyable,frame,k=2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(kp,kpcam, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpcam[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w = latasseavecdesfeaturesincroyable.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    
        frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print ("Not enough matches are found - %d/%d" ) % (len(good),MIN_MATCH_COUNT)
        matchesMask = None  
        
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    
    
    # plt.imshow(img3, 'gray'),plt.show()        
    key = cv2.waitKey(1) & 0xFF;
    if (key == ord('q') or key == ord('x')):
        break;




cv2.waitKey(0)
register.release()
cv2.destroyAllWindows()