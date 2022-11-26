# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,	help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True, help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="", help="optional path to video file")
args = vars(ap.parse_args())
# load the contents of the class labels file, then define the sample
# duration (i.e., # of frames for classification) and sample size
# (i.e., the spatial dimensions of the frame)
CLASSES = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112
# initialize the frames queue used to store a rolling sample duration
# of frames -- this queue will automatically pop out old frames and
# accept new ones
frames = deque(maxlen=SAMPLE_DURATION)
# load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(args["model"])
# grab a pointer to the input video stream
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

# loop over frames from the video stream
while True:
    # read a frame from the video stream
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed then we've reached the end of
    # the video stream so break from the loop
    if not grabbed:
        print("[INFO] no frame read from stream - exiting")
        break
    # resize the frame (to ensure faster processing) and add the
    # frame to our queue
    frame = imutils.resize(frame, width=400)
    frames.append(frame)
    # if our queue is not filled to the sample size, continue back to
    # the top of the loop and continue polling/processing frames
    if len(frames) < SAMPLE_DURATION:
        continue
    # now that our frames array is filled we can construct our blob
    blob = cv2.dnn.blobFromImages(frames, 1.0,
        (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
        swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    # pass the blob through the network to obtain our human activity
    # recognition predictions
    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]
    # draw the predicted activity on the frame
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2)
    # display the frame to our screen
    cv2.imshow("Activity Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
