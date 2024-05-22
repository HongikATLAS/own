import cv2
import time
import socket
import numpy as np
import math

def filt(image):  # filtering 함수
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 흑백으로 color 조정
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
    # canny = cv2.Canny(blur, 100, 300)
    # edge만 추출, 임계값이 클수록 검출이 적게되고 작을수록 검출이 많이됨(실험적으로 픽셀을 잘라내야 함)
    return blur

def hsvExtraction(image, hsvLower, hsvUpper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)
    result = cv2.bitwise_and(image, image, mask=hsv_mask)
    return result


def make_coordinates(image, line_parameters, cross_y):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    try:
        y2 = int(cross_y)  # 교차점까지 표시하도록 변경
    except Exception as error:  # 평행에 가깝게 나올때 오류 나오는 경우 예외처리
        y2 = 0
    x1 = np.clip(int((y1 - intercept) / slope), -10000, 10000)  # x절편 구하는 과정 중 너무 큰 값 나올 시 int 변환중 오류 -10000~ 10000으로 제한
    x2 = np.clip(int((y2 - intercept) / slope), -10000, 10000)
    return np.array([x1, y1, x2, y2])


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
    print(left_fit_average, right_fit_average)
    cross_point_x = (right_fit_average[1] - left_fit_average[1]) / (left_fit_average[0] - right_fit_average[0]) # 교차점 x값
    cross_point_y = (left_fit_average[0] * cross_point_x + left_fit_average[1])  # 교차점 y값
    left_line = make_coordinates(image, left_fit_average, cross_point_y)
    right_line = make_coordinates(image, right_fit_average, cross_point_y)
    print(cross_point_x, cross_point_y, left_line, right_line, 1)
    return np.array([left_line, right_line]), left_save, right_save, cross_point_x, cross_point_y, lines_before, left_line[0],  right_line[0]

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    try:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # 좌표를 잇는 선 그리기 (255,0,0) = 파란색
    except Exception as error:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (0, 480), (640, 480), (255, 0, 0), 10)  # 값이 이상한 경우가 있어서 예외처리
    return line_image


FRAME_WIDTH = 640
FRAME_HEIGTH = 480

cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGTH)

# left_saved=[]    #이전값 저장용 초기값
# right_saved=[]   #이전값 저장용 초기값
# lines_saved=[[]]

while True:
    curr_time = time.time()
    ret, frame = cap.read() #프레임 단위로 읽음
    cv2.imshow("camera", frame)
    gas = filt(frame)
    # cv2.imshow("blur", gas)

    hsvLower = np.array([0, 0, 168])  # 추출할 색의 하한
    hsvUpper = np.array([172, 50, 255])  # 추출할 색의 상한
    hsvResult = hsvExtraction(frame, hsvLower, hsvUpper)
    cv2.imshow('HSV_test1', hsvResult)
    blur = cv2.GaussianBlur(hsvResult, (5, 5), 0)
    # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
    canny = cv2.Canny(blur, 100, 150)
    cv2.imshow("canny", canny)
    print(canny, canny.shape)
    lines = cv2.HoughLinesP(canny, 2, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=30)
    averaged_lines, left_saved, right_saved, cross_x, cross_y, lines_saved, left_location, right_location = average_slope_intercept(
        frame, lines, left_saved, right_saved, lines_saved)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)
    cv2.imshow("line_image", combo_image)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    # time.sleep(0.04)
cap.release()
cv2.destroyAllWindows()