"""
-- ClassID: 74   Based on ID, we should find name
   -- Confidence: 0.853058
   -- Left:    0
   -- Top:     262.78   decimal place we should convert it to pixel
   -- Right:   486.633
   -- Bottom:  476.936
   -- Width:   486.633
   -- Height:  214.156
   -- Area:    104215
   -- Center:  (243.316, 369.858)

"""

import jetson.inference
import jetson.utils

import cv2

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold = 0.5)  #create a network


camSet = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
cap = cv2.VideoCapture(camSet)

while True:
    success, img = cap.read()
    imgCuda = jetson.utils.cudaFromNumpy(img)

    detections = net.Detect(imgCuda) #add detection
    for d in detections:
        #print(d)
        x1,y1,x2,y2 = int(d.Left),int(d.Top),int(d.Right),int(d.Bottom)
        className = net.GetClassDesc(d.ClassID)
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)
        cv2.putText(img,className,(x1+5,y1+15),cv2.FONT_HERSHEY_DUPLEX,0.75, (255,0,255),2)

    #img= jetson.utils.cudaToNumpy(imgCuda) #First method: display detection
    cv2.imshow("Image", img)
    cv2.waitKey(1)
