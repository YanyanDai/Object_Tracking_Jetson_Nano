import jetson.inference
import jetson.utils
import ObjDetMobilenetSSDModule as odmnssdm
import cv2
camSet = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
cap = cv2.VideoCapture(camSet)
myModel = odmnssdm.mnSSD("ssd-mobilenet-v2",0.5)
while True:
    success, img = cap.read()
    objects = myModel.detect(img,True)
    if len(objects)!=0:
        print(objects[0][0])
    cv2.imshow("Image", img)
    cv2.waitKey(1)