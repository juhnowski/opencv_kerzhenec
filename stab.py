# Import numpy and OpenCV
import numpy as np
import cv2
import imutils
import argparse
import datetime

SMOOTHING_RADIUS=100
firstFrame = None
min_area = 100

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  # Define the filter
  f = np.ones(window_size)/window_size
  # Add padding to the boundaries
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
  # Apply convolution
  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  # Remove padding
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed

def smooth(trajectory):
  smoothed_trajectory = np.copy(trajectory)
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

  return smoothed_trajectory
  
    
# Read input video
cap = cv2.VideoCapture('c:/src_video_zap/video/s_y3.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Set up output video
out = cv2.VideoWriter('c:/src_video_zap/video/ss_y3.mp4', fourcc, fps, (w, h))
# Read first frame
_, prev = cap.read()

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32) 

for i in range(n_frames-2):
  # Detect feature points in previous frame
  prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)

  # Read next frame
  success, curr = cap.read()
  if not success:
    break 

  # Convert to grayscale
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

  # Calculate optical flow (i.e. track feature points)
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 

  # Sanity check
  assert prev_pts.shape == curr_pts.shape 

  # Filter only valid points
  idx = np.where(status==1)[0]
  prev_pts = prev_pts[idx]
  curr_pts = curr_pts[idx]

  #Find transformation matrix
  m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less

  # Extract traslation
  dx = m[0,2]
  dy = m[1,2]

  # Extract rotation angle
  da = np.arctan2(m[1,0], m[0,0])

  # Store transformation
  transforms[i] = [dx,dy,da]

  # Move to next frame
  prev_gray = curr_gray

  print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)) + " w="+str(w)+" h="+str(h))

 # Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

smoothed_trajectory = smooth(trajectory)

  # Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory

# Calculate newer transformation array
transforms_smooth = transforms + difference

# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

# Write n_frames-1 transformed frames
for i in range(n_frames-2):
  # Read next frame
  success, frame = cap.read()
  if not success:
    break

  # Extract transformations from the new transformation array
  dx = transforms_smooth[i,0]
  dy = transforms_smooth[i,1]
  da = transforms_smooth[i,2]

  # Reconstruct transformation matrix accordingly to new values
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy

  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))

  # Fix border artifacts
  frame_stabilized = fixBorder(frame_stabilized) 

  # # frame_stabilized = imutils.resize(frame_stabilized, width=500)
  # gray = cv2.cvtColor(frame_stabilized, cv2.COLOR_BGR2GRAY)
  # gray = cv2.GaussianBlur(gray, (21, 21), 0)

  # if firstFrame is None:
  #       firstFrame = gray
  #       continue

  # try:
  #     frameDelta = cv2.absdiff(firstFrame, gray)
  # except Exception:
  #     firstFrame = gray
  #     frameDelta = cv2.absdiff(firstFrame, gray)
      

  # thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

  # thresh = cv2.dilate(thresh, None, iterations=2)
  # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # cnts = imutils.grab_contours(cnts)
	# # loop over the contours
  # for c in cnts:
  #       if cv2.contourArea(c) < min_area:
  #           continue

  #       (x, y, w, h) = cv2.boundingRect(c)
  #       cv2.rectangle(frame_stabilized, (x, y), (x + w, y + h), (0, 255, 0), 2)
  #       text = "Occupied"

  #       cv2.putText(frame_stabilized, "Room Status: {}".format(text), (10, 20),
  #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  #       cv2.putText(frame_stabilized, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
  #       (10, frame_stabilized.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

  #       cv2.imshow("Security Feed", frame_stabilized)


  # Write the frame to the file
  # frame_out = cv2.hconcat([frame, frame_stabilized])
  frame_out = frame_stabilized
  
  # If the image is too big, resize it.
  # if (frame_out.shape[1] > 1920) :
  #   frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)))

  cv2.imshow("Stabilized", frame_out)
  cv2.waitKey(10)
  out.write(frame_out)