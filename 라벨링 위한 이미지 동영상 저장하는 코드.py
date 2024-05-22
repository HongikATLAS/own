import cv2
import time

FRAME_WIDTH = 640
FRAME_HEIGTH = 480

capture = cv2.VideoCapture(cv2.CAP_DSHOW+1)
if capture.isOpened() == False: # 카메라 정상상태 확인

    print(f'Can\'t open the Camera({1})')
    exit()


capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGTH)

file_path = 'C:/Users/kchw3_r2l9a77/OneDrive/Desktop/lane4.mp4' # 동영상 `새로` 갱신할 때마다 번호 바꿔서 저장 / 동영상 저장하는 파일 따로 만들어야 됨

# Define the codec and create VideoWriter object. The output is stored in 'output.mp4' file.
out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (FRAME_WIDTH, FRAME_HEIGTH))

prev_time = 0
total_frames = 0
start_time = time.time()
while cv2.waitKey(1) < 0:
    curr_time = time.time()

    ret, frame = capture.read()
    total_frames = total_frames + 1

    # Write the frame into the file (VideoWriter)
    out.write(frame)

    term = curr_time - prev_time
    fps = 1 / term
    prev_time = curr_time
    fps_string = f'FPS = {fps:.2f}'
    print(fps_string)

    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(10) & 0xFF== ord('x'):
        capture.release()
        cv2.destroyAllWindows()
        break
end_time =time.time()
fps = total_frames / (start_time - end_time)
print(f'total_frames = {total_frames},  avg FPS = {fps:.2f}')

capture.release()
out.release()
cv2.destroyAllWindows()