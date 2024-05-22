import cv2
import time
import socket
import numpy as np
import math




def make_coordinates(image, line_parameters, cross_y):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    try:
        y2 = int(cross_y)  # 교차점까지 표시하도록 변경
    except Exception as error:  # 평행에 가깝게 나올때 오류 나오는 경우 예외처리
        y2 = 0
    x1 = np.clip(int((y1 - intercept) / slope), -10000, 10000)  # x절편 구하는 과정 중 너무 큰 값 나올 시 int 변환중 오류 -10000~ 10000으로 제한
    x2 = np.clip(int((y2 - intercept) / slope), -10000, 10000)
    return np.array([x1, y1, x2, y2], dtype=int)


def average_slope_intercept(image, line_a, left_before, right_before, lines_before):
    left_fit = []
    right_fit = []
    try:
        for line in line_a:
            # print(line, 100, type(line), line.shape)
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            x_bottom = ((480 - y1)/slope)+x1
            if slope < 0 and x_bottom < 320:
                left_fit.append((slope, intercept))
            elif slope > 0 and x_bottom > 320:  # 중앙을 벗어난 오른쪽 차선도 좌측 차선으로 인식할 수 있으므로 같은 방식으로 제한
                right_fit.append((slope, intercept))
                right_fit.append((slope, intercept))
        lines_before = line_a
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
    # print(left_fit_average, right_fit_average)
    print(left_fit_average, right_fit_average, 10)
    cross_point_x = (right_fit_average[1] - left_fit_average[1]) / (left_fit_average[0] - right_fit_average[0]) # 교차점 x값
    cross_point_y = (left_fit_average[0] * cross_point_x + left_fit_average[1])  # 교차점 y값
    left_line = make_coordinates(image, left_fit_average, cross_point_y)
    right_line = make_coordinates(image, right_fit_average, cross_point_y)
    # print(cross_point_x, cross_point_y, left_line, right_line, left_line[0], right_line[1], 1)
    return np.array([left_line, right_line]), left_save, right_save, cross_point_x, cross_point_y, lines_before, left_line[0],  right_line[0], left_line, right_line


def filt(image):  # filtering 함수
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 흑백으로 color 조정
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
    canny = cv2.Canny(blur, 100, 300)
    # edge만 추출, 임계값이 클수록 검출이 적게되고 작을수록 검출이 많이됨(실험적으로 픽셀을 잘라내야 함)
    return canny

# def filt(image):  # filtering 함수
#     blur = cv2.GaussianBlur(image, (5, 5), 0)
#     img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#     # clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(40, 40))
#     # img[:, :, 0] = clahe.apply(img[:, :, 0])
#     # img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
#     # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 흑백으로 color 조정
#     # _, white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#     yellow = cv2.inRange(img, (0, 128, 0), (100, 255, 100))
#     # add = cv2.bitwise_or(white, yellow)
#     canny = cv2.Canny(yellow, 250, 300)
#     return canny

def ROI_lines(ROI):  # polygons(ROI 점들) 수정하면 자동으로 직선 4개 만들기
    return np.array([np.concatenate((ROI[0][0], ROI[0][3]), axis=0), np.concatenate((ROI[0][3], ROI[0][2]), axis=0),
                     np.concatenate((ROI[0][2], ROI[0][1]), axis=0), np.concatenate((ROI[0][1], ROI[0][0]), axis=0)])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    try:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # 좌표를 잇는 선 그리기 (255,0,0) = 파란색
    except Exception as error:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (0, 480), (640, 480), (255, 0, 0), 10)  # 값이 이상한 경우가 있어서 예외처리
    return line_image

def display_ROI(image, lines):  # ROI 빨간선으로 표시하기 위해 따로 생성
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 좌표를 잇는 선 그리기 (0,0,255) = 빨간색
    return line_image

def display_ROI_first(image, lines):  # 초기에 차선 감지 위해 지정한 영역
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 좌표를 잇는 선 그리기 (0,0,255) = 빨간색
    return line_image

def ROI(image, Roisize):
    mask = np.zeros_like(image)  # image와 같은 shape의 0배열 만듬
    cv2.fillPoly(mask, Roisize, 255)  # 다각형 그리기(image의 mask를 위한것)
    masked_image = cv2.bitwise_and(image, mask)  # image와 mask의 겹치는(and) 이미지 출력
    # cv2.imshow("masked_image", masked_image)
    return masked_image

def calculate(x_c, y_c, mid_line):                 #세타, 횡방향 오차 e 구하는 함수 result
    deg = math.degrees(math.atan2(x_c - mid_line, 480 - y_c))             #세타
    # err = (x_c-320) * 1.8 * math.cos(math.radians(deg)) - 510 * math.sin(math.radians(deg))     #횡방향 오차
    err = x_c - mid_line# x좌표가 320이 아니고 양쪽 차선들의 y=480과 만나는 x좌표들의 중점이 mid_line이다.
    return deg, err

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

left_saved=[]    #이전값 저장용 초기값
right_saved=[]   #이전값 저장용 초기값
lines_saved=[[]]


cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
polygons = np.array([[(0, 300),(0,400), (640, 400), (640, 300)]])

prev_time = 0
total_frames = 0
start_time = time.time()
camera_state = 1

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
            print(fps, 234234234234)

            prev_time = curr_time
            fps_string = f'FPS = {fps:.2f}'
            # cv2.imshow("first", frame)
            # cv2.imshow("second", second_frame)
            ret, frame = cap.read() #프레임 단위로 읽음
            difference = cv2.absdiff(frame, second_frame, dst=None)

            # canny_image = filt(frame)
            # cropped_image = ROI(canny_image, polygons)
            # lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, np.array([]), minLineLength=10, maxLineGap=5)
            # lines = np.float32(lines)
            # averaged_lines, left_saved, right_saved, cross_x, cross_y, lines_saved, left_location, right_location, left_line, right_line = average_slope_intercept(frame, lines, left_saved, right_saved, lines_saved)
            # line_image = display_lines(frame, averaged_lines)
            # combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)
            # # cv2.imshow("combo_image", combo_image)
            # # print(left_line, right_line, 100) #len(left_line) = 4
            # left_roi_center = (350-left_line[1])*((left_line[0]-left_line[2])/(left_line[1]-left_line[3])) + left_line[0]
            # right_roi_center = (350-right_line[1])*((right_line[0]-right_line[2])/(right_line[1]-right_line[3])) + right_line[0]
            # print(left_line, right_line, left_roi_center, right_roi_center, 100)
            # left_polygons = np.array([[(left_roi_center - 50, 300),(left_roi_center - 50,400), (left_roi_center + 50, 400), (left_roi_center + 50, 300)]], dtype = int)
            # right_polygons = np.array([[(right_roi_center - 50, 300), (right_roi_center - 50, 400), (right_roi_center + 50, 400),(right_roi_center + 50, 300)]], dtype=int)
            # ROI_right_image = display_ROI(frame, ROI_lines(right_polygons))
            # ROI_left_image = display_ROI(frame, ROI_lines(left_polygons))
            # combo_image2 = cv2.addWeighted(combo_image, 1, ROI_right_image, 1, 1)
            # combo_image2 = cv2.addWeighted(combo_image2, 1, ROI_left_image, 1, 1)
            # ROI_total = display_ROI_first(frame, ROI_lines(polygons))
            # combo_image2 = cv2.addWeighted(combo_image2, 1, ROI_total, 1, 1)
            # total_result = cv2.hconcat([frame, second_frame]).copy()
            # total_result = cv2.resize(total_result, (0, 0), fx=0.5, fy=0.5)
            # cv2.imshow("add_roi", combo_image2)
            # cv2.imshow("total_result", total_result)
            # if len(str(left_location)) and len(str(right_location))
            cv2.imshow("Dd", difference)


            if cv2.waitKey(1) & 0xff == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()




