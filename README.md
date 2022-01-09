# Yolo

This script is based on the Yolo series. Since it is quite complicated for beginers to understand the layers output and structure of the network, this script is written as a package for beginers or developers that are in rush to test their model.

The weight file and config file in the Demo file are downloaded from https://pjreddie.com/darknet/yolo/ 

By using the testcuda script you could found out if your cuda and cudnn is install correctly. If you cude is installed, you can also enable you gpu in the detection process by simply typing


````
Net.Init(cuda = True)
````



**This is what you could expect within 24 lines of codes**

![image](https://github.com/jj98-hub/Yolo/blob/26cdbba7c40198388f1417423329e80b16a54da8/DemoOutput.gif)

Try it out and have fun!
