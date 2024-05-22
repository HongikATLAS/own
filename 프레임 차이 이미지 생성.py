import cv2
import time
import socket
import numpy as np
import math

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
polygons = np.array([[(0, 300),(0,400), (640, 400), (640, 300)]])

prev_time = 0
total_frames = 0
start_time = time.time()
camera_state = 1

ret, old_frame = cap.read()


while True:
    for i in range(2):
        rat, frame = cap.read()
        if i == 0:
            frame = frame
        elif i == 1:
            second_frame = frame
            curr_time = time.time()
            ret, frame = cap.read()  # 프레임 단위로 읽음
            total_frames = total_frames + 1

            term = curr_time - prev_time
            fps = 1 / term
            print(fps)

            prev_time = curr_time
            fps_string = f'FPS = {fps:.2f}'
            # cv2.imshow("first", frame)
            # cv2.imshow("second", second_frame)
            ret, frame = cap.read() #프레임 단위로 읽음
            difference = cv2.absdiff(frame, second_frame, dst=None)
            print(difference)
            cv2.imshow("frame", frame)
            cv2.imshow("second", second_frame)
            cv2.imshow("Dd", difference)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
