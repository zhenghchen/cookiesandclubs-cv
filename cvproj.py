import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        if isinstance(posCenter, list):
            self.posCenter = posCenter
        else:
            self.posCenter = [posCenter[0], posCenter[1]]
        self.targetPos = self.posCenter.copy() 
        self.size = size
        self.smoothing = 0.5 

    def isInBounds(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        return (cx - w // 2 < cursor[0] < cx + w // 2 and 
                cy - h // 2 < cursor[1] < cy + h // 2)

    def update(self, cursor, isPinching, activeRect):
        if activeRect == self and isPinching:
            self.targetPos = [cursor[0], cursor[1]]
            
            self.posCenter[0] += (self.targetPos[0] - self.posCenter[0]) * (1 - self.smoothing)
            self.posCenter[1] += (self.targetPos[1] - self.posCenter[1]) * (1 - self.smoothing)
            return self
        
        if activeRect is None:
            
            if isPinching and self.isInBounds(cursor):
                
                self.targetPos = [cursor[0], cursor[1]]
                
                self.posCenter[0] += (self.targetPos[0] - self.posCenter[0]) * 0.5
                self.posCenter[1] += (self.targetPos[1] - self.posCenter[1]) * 0.5
                return self
        elif activeRect != self:
            return activeRect
        
        if not isPinching and activeRect == self:
            self.targetPos = self.posCenter.copy() 
            return None
            
        return activeRect


rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

activeRect = None
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    cursor = None
    isPinching = False

    if hands:
        hand1 = hands[0]
        
        point1 = hand1["lmList"][4][:2]   # Thumb tip (x,y)
        point2 = hand1["lmList"][8][:2]   # Index finger tip (x,y)
        distance = detector.findDistance(point1, point2, img)[0] 
        
        cursor = point2 
        isPinching = distance < 40

    if not hands or not isPinching:
        activeRect = None
    
    if cursor is not None:
        for rect in rectList:
            activeRect = rect.update(cursor, isPinching, activeRect)

    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = map(int, rect.posCenter)  
        w, h = rect.size
        pt1 = (int(cx - w // 2), int(cy - h // 2))
        pt2 = (int(cx + w // 2), int(cy + h // 2))
        cv2.rectangle(imgNew, pt1, pt2, colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (pt1[0], pt1[1], w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)