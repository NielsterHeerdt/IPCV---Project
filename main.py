import cv2
import numpy as np
import os

def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame, lines



video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_segment = int(fps * 2.5)  
total_segments = 8
total_frames = frames_per_segment * total_segments

while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    # Step 1: corner detection
    frame, lines = detect_lines(frame)
    cv2.imshow('frame',frame)
    cv2.waitKey(25)
    frame_count += 1 
cap.release()
cv2.destroyAllWindows()