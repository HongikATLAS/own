import cv2
import time
import matplotlib.pyplot as plt
import socket
import numpy as np


def ROI_lines(ROI):  # polygons(ROI 점들) 수정하면 자동으로 직선 4개 만들기
    return np.array([np.concatenate((ROI[0][0], ROI[0][3]), axis=0), np.concatenate((ROI[0][3], ROI[0][2]), axis=0),
                     np.concatenate((ROI[0][2], ROI[0][1]), axis=0), np.concatenate((ROI[0][1], ROI[0][0]), axis=0)])


def make_coordinates(image, line_parameters, cross_y):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(cross_y)  # 교차점까지 표시하도록 변경
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, line_a, left_before, right_before, lines_before):
    left_fit = []
    right_fit = []
    try:
        for line in line_a:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        lines_before = lines
    except Exception as error:
        line_a = lines_before  # 좌측 차선 이전값 불러오기
        for line in line_a:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))


    if len(left_fit) == 0:
        left_fit = left_before  # 좌측 차선 이전값 불러오기

    if len(right_fit) == 0:
        right_fit = right_before  # 우측 차선 이전값 불러오기

    left_save = left_fit  # 좌측 차선 저장
    right_save = right_fit  # 우측 차선 저장


    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    cross_point_x = (right_fit_average[1] - left_fit_average[1]) / (left_fit_average[0] - right_fit_average[0]) # 교차점 x값
    cross_point_y = (left_fit_average[0] * cross_point_x + left_fit_average[1])  # 교차점 y값
    left_line = make_coordinates(image, left_fit_average, cross_point_y)
    right_line = make_coordinates(image, right_fit_average, cross_point_y)
    #print(left_line, right_line)
    return np.array([left_line, right_line]), left_save, right_save, cross_point_x, cross_point_y, lines_before, left_line[0],  right_line[0]

def auto_canny(image, sigma=0.6):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged



# def filt(image):  # filtering 함수
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)  # 흑백으로 color 조정
#     blur = cv2.GaussianBlur(img, (5, 5), 0)
#     # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
#     Lower = np.array([0, 0, 150])  # 0,0,150 #20 0 90
#     Upper = np.array([150, 150, 255])  # 150,150,255 #30 255 255
#     yellow = cv2.inRange(blur, Lower, Upper)
#     # canny = auto_canny(yellow)
#     return yellow

# def filt(image):  # filtering 함수
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 흑백으로 color 조정
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)
#     # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
#     canny = cv2.Canny(blur, 150, 250)
#     # edge만 추출, 임계값이 클수록 검출이 적게되고 작을수록 검출이 많이됨(실험적으로 픽셀을 잘라내야 함)
#     return canny

def filt(image):  # filtering 함수
    img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(40, 40))
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 흑백으로 color 조정
    _, white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    Lower = np.array([20,0,90])
    Upper = np.array([150,150,255])
    yellow = cv2.inRange(blur,Lower,Upper)
    add = cv2.bitwise_or(white, yellow)
    canny = auto_canny(add)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # 좌표를 잇는 선 그리기 (255,0,0) = 파란색
    return line_image


def display_ROI(image, lines):  # ROI 빨간선으로 표시하기 위해 따로 생성
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 좌표를 잇는 선 그리기 (0,0,255) = 빨간색
    return line_image


def ROI(image, Roisize):
    mask = np.zeros_like(image)  # image와 같은 shape의 0배열 만듬
    cv2.fillPoly(mask, Roisize, 255)  # 다각형 그리기(image의 mask를 위한것)
    masked_image = cv2.bitwise_and(image, mask)  # image와 mask의 겹치는(and) 이미지 출력
    return masked_image

FRAME_WIDTH = 640
FRAME_HEIGTH = 480

cap = cv2.VideoCapture('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/test.mp4')#1, cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
polygons = np.array([[(0, 0), (0, 480), (640, 480), (640, 0)]])    #ROI 값 변경하기 위해 바깥으로 뺐음

left_saved=[]    #이전값 저장용 초기값
right_saved=[]   #이전값 저장용 초기값
lines_saved=[[]]


# Define the codec and create VideoWriter object. The output is stored in 'output.mp4' file.

prev_time = 0
total_frames = 0
start_time = time.time()

while True:
    curr_time = time.time()
    ret, frame = cap.read() #프레임 단위로 읽음
    total_frames = total_frames + 1
    #
    # term = curr_time - prev_time
    # fps = 1 / term
    # prev_time = curr_time
    # fps_string = f'FPS = {fps:.2f}'
    # print(fps_string)
    #
    # cv2.putText(frame, fps_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

    if not ret:                         #카메라로부터 이미지 못 받아올 경우 camera_off 불리언 True, 다시 위에서부터 실행
        camera_off = True
        print(camera_off)
        continue

    canny_image = filt(frame)
    cropped_image = ROI(canny_image, polygons)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, np.array([]), minLineLength=90, maxLineGap=70)
    averaged_lines, left_saved, right_saved, cross_x,cross_y, lines_saved, left_location, right_location = average_slope_intercept(frame, lines, left_saved, right_saved, lines_saved)
    line_image = display_lines(frame, lines)
    ROI_image = display_ROI(frame, ROI_lines(polygons))                       #ROI 영역 이미지 생성
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # combo_image2 = cv2.addWeighted(combo_image, 1, ROI_image, 1, 1)   #ROI 이미지 합치기
    cv2.imshow("aa", line_image)
    cv2.imshow("result", ROI_image)

    # # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # # sock.connect(('localhost', 6340))
    # cross_x = np.round(cross_x, 0)
    # cross_y = np.round(cross_x, 0)
    # left_location = np.round(left_location, 0)
    # right_location = np.round(right_location, 0)
    # print(cross_x, cross_y, left_location, right_location)
    # # msg = str(cross_x)
    # msg1 = str(left_location)
    # msg2 = str(right_location)
    # sock.send(msg.encode("utf-8"))
    # sock.send(msg1.encode("utf-8"))
    # sock.send(msg2.encode("utf-8")) # msg2를 유니코드로 인코딩 한 뒤에 sock을 통해 보내겠다는 의미이다.

    # msg_raw = sock.recv(20)
    # msg = msg_raw.decode("utf-8", "ignore") # decode는 일반 바이트 문자열에서 유니 코드로 변환하기 위해 수행하는 작업
    # msg1_raw = sock.recv(20)
    # msg1 = msg1_raw.decode("utf-8", "ignore")
    # msg2_raw = sock.recv(20)
    # msg2 = msg2_raw.decode("utf-8", "ignore") 근데 이 코드가 없어도 잘 돌아간다. / 보낸 거 다시 받는 거라 크게 의미가 없는 것 같다.

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
