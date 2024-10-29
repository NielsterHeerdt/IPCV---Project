import cv2
import numpy as np
import os

# function to detect lines
def detect_lines(frame,boundaries):
    intersections = []

    # remove colors of lower values (the lines are white)
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 9, 3, 0.01) 
    corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dilated = cv2.dilate (thresh, None, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            intersections.append(contour)
    return frame, intersections


# initiate video 
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# defenition of parameters
frame_count = 0
frames_per_segment = int(fps * 2.5)  
total_segments = 8
total_frames = frames_per_segment * total_segments
boundaries = [([170, 170, 100], [255, 255, 255])] # define threshold to only use the white lines 



while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    # Step 1: corner/line detection (localize calibration points)
    frame, intersections = detect_lines(frame,boundaries)

    # Step 2 Intrinsic camera calibration using the known 3D positions of reference objects

    # Step 3 External camera calibration: the pose of the camera relative to the 3D reference objects

    # Step 4 Tracking of 2D points and/or lines in the movie.

    # Step 5 Based on these tracked points and/or lines, camera pose tracking during the movie.

    # Step 6 Projection of a virtual banner which has a rectangular shape in the real world and located near a court line


    cv2.imshow('frame',frame)
    cv2.waitKey(25)
    frame_count += 1 
cap.release()
cv2.destroyAllWindows()