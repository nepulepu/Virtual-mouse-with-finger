import cv2
import numpy as np
import HandTrackingModule as htm
import time

camW,camH= 640,480

cap=cv2.VideoCapture(0)
cap.set(3,camW)
cap.set(4,camH)
pTime=0
detector= htm.handDetector(maxHands=1)
while True:
    # 1.Find hand landmarks
    success,img=cap.read()
    img= detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. get tip of index and mid fing
    # 3. Check which fingers up
    # 4. Only index finger : moving
    # 5. Convert Coordinates
    # 6. Smoothen Values
    # 7. Move mouse
    # 8. if both index and middle fingers are up: clicking mode
    # 9. Finde distance between fingers
    # 10. Click mouse if distance short(touching)
    # 11. frame rate
    cTime=time.time()
    fps= 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5)
    # 12. Display
    cv2.imshow("Image",img)
    cv2.waitKey(1)