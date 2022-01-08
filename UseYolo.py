import cv2
import numpy as np
import time

class YoloDetector:
    
    net = None
    confidenceThreshold = 0.5
    nmsThreshold = 0.3 #lower this value less bbox will be found

    classesFile = None  #file that contains all the classes name
    classNames = []


    #Below loads the configuration and weight files
    modelConfig = None
    modelWeight = None

    #Below Initialize the Darknet-opencv and set the target to use CPU
    def NetInit(self,cuda = False):
        if self.classesFile != None:
            with open (self.classesFile,"rt") as f:
                self.classNames = f.read().rstrip("\n").split("\n")
    
        if self.modelConfig and self.modelWeight != None:
            self.net = cv2.dnn.readNetFromDarknet(self.modelConfig,self.modelWeight)
            if cuda == False:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("CUDA not activated, using CPU") 
            elif cuda ==True:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("CUDA activated, using GPU")
            else:
                print("Please Set cuda to TRUE or FALSE ONLY!")
        else:
            print("No Weights and Configuration File is Loaded") 
        
    
    def SendImg2Net (self,img,wTarget = 320,hTarget = 320):
        if self.net !=None:
            blob = cv2.dnn.blobFromImage(img,1/255,(wTarget,hTarget),
            [0,0,0],1,crop = False)  #chage to image to blob format since the network only takes blob input
            self.net.setInput(blob)
            layerNames = self.net.getLayerNames()
            outputNames = [layerNames[i[0]-1] for i in self.net.getUnconnectedOutLayers()] #since start from 1 so we need i[0]-1
            return self.net.forward(outputNames)
        else:
            print("Initiate network with NetInit before using this function")

    
    def FindObjects(self,outputs,img,draw = True,ret = False):
        hT, wT, cT = img.shape
        bboxs = []
        classIds = []
        confidence = []
        retlist = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores) #return the index of max value
                confi = scores[classId] #extract the max value
                if confi > self.confidenceThreshold:
                    w,h = int(det[2]*wT), int(det[3]*hT) #the third and forth element in det is width and height of bounding box 
                    x,y = int(det[0]*wT-w/2), int(det[1]*hT-h/2)  #the first and secodn element in det is center x positio and center y position 
                    bboxs.append([x,y,w,h])
                    classIds.append(classId)
                    confidence.append(float(confi))
        indices = cv2.dnn.NMSBoxes(bboxs,confidence,self.confidenceThreshold,self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = bboxs[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            retlist.append({'Class':self.classNames[classIds[i]].upper(),'BoundingBox':box,'Confidence':confidence[i]}) #BoundingBox will return its top left corner position x,y and width and height of the box
            if draw:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
                cv2.putText(img,f'{self.classNames[classIds[i]].upper()} {int(confidence[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),2)
        if ret == True:
            return retlist
            
            

if __name__ == '__main__':

    # define yolo as a YoloDetector Object
    yolo = YoloDetector()
    yolo.modelConfig = "yoloDemo/yolov3.cfg"
    yolo.modelWeight = "yoloDemo/yolov3.weights"
    yolo.classesFile = "yoloDEMO/coco.names"
    yolo.NetInit()

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

