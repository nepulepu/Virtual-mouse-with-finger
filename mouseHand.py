import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

camW,camH= 640,480
frameR=100
smoothen=5

pTime=0
prevX,prevY=0,0
curX,curY=0,0

cap=cv2.VideoCapture(0)
cap.set(3,camW)
cap.set(4,camH)
detector= htm.handDetector(maxHands=1)
scrnW,scrnH=autopy.screen.size()
print(scrnW,scrnH)
while True:
    # 1.Find hand landmarks
    success,img=cap.read()
    img= detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    

    # 2. get tip of index and mid fing
    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]
        # print(x1,y1,x2,y2)

        # 3. Check which fingers up
        fingers=detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img,(frameR,frameR),(camW-frameR,camH-frameR),(255,0,255),2)

        # 4. Only index finger : moving mode
        if (fingers[1]==1 and fingers[2]==0):

            # 5. Convert Coordinates
            
            x3=np.interp(x1,(frameR,camW-frameR),(0,scrnW))
            y3=np.interp(y1,(frameR,camH-frameR),(0,scrnH))

            # 6. Smoothen Values
            curX=prevX+(x3-prevX)/smoothen
            curY=prevY+(y3-prevY)/smoothen

            # 7. Move mouse
            autopy.mouse.move(scrnW-curX,curY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            prevX,prevY=curX,curY
        
        # 8. if both index and middle fingers are up: clicking mode
        if (fingers[1]==1 and fingers[2]==1):
            # 9. Finde distance between fingers
            leng,img, lnInfo =detector.findDistance(8,12,img)
            # 10. Click mouse if distance short(touching)
            if (leng <40):
                cv2.circle(img,(lnInfo[4],lnInfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

    # 11. frame rate
    cTime=time.time()
    fps= 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5)
    # 12. Display
    cv2.imshow("Image",img)
    cv2.waitKey(1)