import jetson.inference
import jetson.utils
import cv2
import ObjDetMobilenetSSDModule as odmnssdm
import SerialModule as sm 
import numpy as np 
import time

# frameWidth = 640
# frameHeight = 480

camSet = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

cap = cv2.VideoCapture(camSet)
ser = sm.initconnection('/dev/ttyACM0', 115200)

perrorLR, perrorUD = 0,0
myModel = odmnssdm.mnSSD("ssd-mobilenet-v2",0.5)


def trackObject(cx,cy,w,h,OHeight):
    global perrorLR, perrorUD
    kLR = [1, 0.1]
   
    #Left and Right
    errorLR = w//2 - cx
    posTurn = kLR[0]*errorLR + kLR[1]*(errorLR-perrorLR)
    posTurn = int(np.interp(posTurn,[-w//2,w//2],[100,-100]))
    perrorLR = errorLR
    #Forward and Stop
    if OHeight >= 0.5*h:
        posSpeed = 0
    elif OHeight>= 0.3*h and OHeight<0.05*h:
        posSpeed = 40
    else:
        posSpeed = 50

    sm.sendData(ser,[posSpeed,posTurn],4)

while True:
    success, img = cap.read()
    objects = myModel.detect(img,True)
    if len(objects)!=0:
        for obj in objects:
            if obj[0]=='mouse':
                cx = int(obj[1].Center[0])
                cy = int(obj[1].Center[1])
                OHeight = int(obj[1].Bottom-obj[1].Top)
                trackObject(cx,cy,img.shape[1],img.shape[0],OHeight)
            else: sm.sendData(ser,[0,0],4)
    else:
        sm.sendData(ser,[0,0],4)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sm.sendData(ser,[0,0],4)
        break