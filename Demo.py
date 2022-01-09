from UseYolo import YoloDetector
import cv2
import numpy as np
import time

# define yolo as a YoloDetector Object
yolo = YoloDetector()
yolo.modelConfig = "yoloDemo/yolov3.cfg"
yolo.modelWeight = "yoloDemo/yolov3.weights"
yolo.classesFile = "yoloDEMO/coco.names"
yolo.NetInit(cuda=True)

cap = cv2.VideoCapture(0)

while True:
    stime = time.time()
    ret, img = cap.read()
    LayerOutputs = yolo.SendImg2Net(img)
    Data = yolo.FindObjects(LayerOutputs,img,ret=True)
    print(Data)
    fps = int(1/(time.time()-stime))
    cv2.putText(img,"FPS : "+str(fps),(10,15),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),1)
    cv2.imshow("Image",img)
    cv2.waitKey(1)