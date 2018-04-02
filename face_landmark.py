#This code is copied from https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
# built this script only for hands on practice 

from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2 
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
    help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


frame = cv2.imread('fig1.png') #fig1.png') # vs.read()
frame = imutils.resize(frame, width=400)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale frame
rects = detector(gray, 0)

# loop over the face detections
while True:
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        
    # show the frame
    key = cv2.waitKey(1) & 0xFF
    # print(key)
    if key == ord("q"):
        break
    cv2.imshow("Frame", frame)
     

jaw_line = shape[6:9]
x =  [i[0] for i in jaw_line] # X co ordinates of jaw line
y =  [i[1] for i in jaw_line] # Y co ordinates of jaw line


# gradiant is defined as sum( (x(i) - mean(x)) * (y(i) - mean(y) ) / sum( (x(i) - mean(x))**2 )
num = 0
den = 0 
for i in range(len(jaw_line)): 
    num += (x[i]-np.mean(x))*(y[i]-np.mean(y))
    den += (x[i]-np.mean(x))**2
 


print("Attribute 22: Jaw gradient =",np.abs(num/den)) 


 

