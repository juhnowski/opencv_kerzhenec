import numpy as np
import cv2 as cv
import imutils
import argparse
import datetime


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
# ap.add_argument("-a", "--min-area", type=int, default=10, help="minimum area size")
args = vars(ap.parse_args())

min_area = 300

cap = cv.VideoCapture('video/y3.mp4')
fps = cap.get(cv.CAP_PROP_FPS)

# Get width and height of video stream
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter('c:/src_video_zap/video/res_y3.mp4', fourcc, fps, (w, h))

firstFrame = None
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = imutils.resize(frame, width=500)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    frameDelta = cv.absdiff(firstFrame, gray)
    thresh = cv.threshold(frameDelta, 60, 255, cv.THRESH_BINARY)[1]

    thresh = cv.dilate(thresh, None, iterations=2)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
	# loop over the contours
    for c in cnts:
        if cv.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

        cv.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        cv.imshow("Security Feed", frame)

        # cv.imshow("Thresh", thresh)
        # cv.imshow("Frame Delta", frameDelta)

    # cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
    out.write(frame)
cap.release()
cv.destroyAllWindows()

 