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

class mnSSD():
    def __init__(self, path, threshold):
        self.path = path
        self.threshold = threshold
        self.net = jetson.inference.detectNet(self.path, self.threshold)  #create a network
    def detect(self, img, display=False):
        imgCuda = jetson.utils.cudaFromNumpy(img)
        detections = self.net.Detect(imgCuda,overlay = "OVERLAY_NONE") #add detection
        objects = []
        for d in detections:
            className = self.net.GetClassDesc(d.ClassID)
            objects.append([className,d])
            if display:
                x1,y1,x2,y2 = int(d.Left),int(d.Top),int(d.Right),int(d.Bottom)   
                cx,cy = int(d.Center[0]), int(d.Center[1])         
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.circle(img,(cx,cy),5,(255,0,0),2,cv2.FILLED)
                cv2.line(img,(x1,cy),(x2,cy),(255,0,0),1)
                cv2.line(img,(cx,y1),(cx,y2),(255,0,0),1)
                cv2.putText(img,className,(x1+5,y1+15),cv2.FONT_HERSHEY_DUPLEX,0.75, (255,0,0),2)
                cv2.putText(img,f'FPS:{int(self.net.GetNetworkFPS())}',(30,30),cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2)
        return objects

def main():
    camSet = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    cap = cv2.VideoCapture(camSet)
    myModel = mnSSD("ssd-mobilenet-v2",0.5)
    while True:
        success, img = cap.read()
        objects = myModel.detect(img,True)
        if len(objects)!=0:
            print(objects[0][0])
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()