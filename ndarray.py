import cv2
import time
import socket
import numpy as np
import math

cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0
total_frames = 0
start_time = time.time()
camera_state = 1

while True:
    curr_time = time.time()
    ret, frame = cap.read()  # 프레임 단위로 읽음
    total_frames = total_frames + 1

    term = curr_time - prev_time
    fps = 1 / term
    print(fps, 234234234234)
    prev_time = curr_time
    fps_string = f'FPS = {fps:.2f}'
    rat, frame = cap.read()
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

