import cv2
import time
import socket
import numpy as np
import math
#
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(('localhost', 6340))


def ROI_lines(ROI):  # polygons(ROI 점들) 수정하면 자동으로 직선 4개 만들기
    return np.array([np.concatenate((ROI[0][0], ROI[0][3]), axis=0), np.concatenate((ROI[0][3], ROI[0][2]), axis=0),
                     np.concatenate((ROI[0][2], ROI[0][1]), axis=0), np.concatenate((ROI[0][1], ROI[0][0]), axis=0)])


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
    return np.array([left_line, right_line]), left_save, right_save, cross_point_x, cross_point_y, lines_before, left_line[0],  right_line[0], left_line, right_line


def filt(image):  # filtering 함수
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 흑백으로 color 조정
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
    canny = cv2.Canny(blur, 250, 300)
    # edge만 추출, 임계값이 클수록 검출이 적게되고 작을수록 검출이 많이됨(실험적으로 픽셀을 잘라내야 함)
    return canny

# def filt(image):
#     # HSV로 색추출
#     hsvLower = np.array([0, 0, 168])    # 추출할 색의 하한
#     hsvLower = np.array([0, 0, 168])    # 추출할 색의 하한
#     hsvUpper = np.array([172, 50, 255])    # 추출할 색의 상한
#     hsvResult = hsvExtraction(image, hsvLower, hsvUpper)
#     print(hsvResult, hsvResult.shape)
#
#     # cv2.imshow('HSV_test1', hsvResult)
#     cv2.imshow("image", image)
#     blur = cv2.GaussianBlur(hsvResult, (5, 5), 0)
#     # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
#     canny = cv2.Canny(blur, 100, 300)
#     return canny

# def hsvExtraction(image, hsvLower, hsvUpper):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)
#     result = cv2.bitwise_and(image, image, mask=hsv_mask)
#     return result

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
    cv2.imshow("masked_image", masked_image)
    return masked_image

def calculate(x_c, y_c, mid_line):                 #세타, 횡방향 오차 e 구하는 함수 result
    deg = math.degrees(math.atan2(x_c - mid_line, 480 - y_c))             #세타
    # err = (x_c-320) * 1.8 * math.cos(math.radians(deg)) - 510 * math.sin(math.radians(deg))     #횡방향 오차
    err = x_c - mid_line# x좌표가 320이 아니고 양쪽 차선들의 y=480과 만나는 x좌표들의 중점이 mid_line이다.
    return deg, err


FRAME_WIDTH = 640
FRAME_HEIGTH = 480

cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGTH)
polygons = np.array([[(0, 120), (0, 480), (640, 480), (640, 120)]])     #ROI 값 변경하기 위해 바깥으로 뺐음

left_saved=[]    #이전값 저장용 초기값
right_saved=[]   #이전값 저장용 초기값
lines_saved=[[]]


# Define the codec and create VideoWriter object. The output is stored in 'output.mp4' file.

prev_time = 0
total_frames = 0
start_time = time.time()
camera_state = 1
while True:
    curr_time = time.time()
    ret, frame = cap.read() #프레임 단위로 읽음
    total_frames = total_frames + 1

    term = curr_time - prev_time
    fps = 1 / term
    prev_time = curr_time
    fps_string = f'FPS = {fps:.2f}'

    cv2.putText(frame, fps_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

    # if not ret:                         #카메라로부터 이미지 못 받아올 경우 camera_off 불리언 True, 다시 위에서부터 실행
    #     camera_off = True
    #     print(camera_off)
    #     continue

    pts1 = np.float32([[200, 0], [50, 360], [590, 360], [440, 0]])
    pts2 = np.float32([[0, 0], [0, 480], [640, 480], [640, 0]])
    M = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    dst = cv2.warpPerspective(frame , M, (640, 480)) # dst는 버드아이뷰 전환한 화면
    canny_image = filt(frame)
    canny_bird_image = filt(dst)
    cropped_image = ROI(canny_image, polygons)          #error 날 일 없음
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 20, np.array([]), minLineLength=10, maxLineGap=5)
    # minlinelength가 너무 낮으면 너무 짧은 선이 검출된다 / max_line_gap보다 점 사이의 거리가 크면 나와 다른 선
    lines = np.float32(lines)
    print(lines, lines.shape, 150)
    averaged_lines, left_saved, right_saved, cross_x, cross_y, lines_saved, left_location, right_location, left_line, right_line = average_slope_intercept(frame, lines, left_saved, right_saved, lines_saved)
    line_image = display_lines(frame, averaged_lines)
    ROI_image = display_ROI(frame, ROI_lines(polygons))                       #ROI 영역 이미지 생성
    combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)
    # left_roi_center = (350 - left_line[1]) * ((left_line[0] - left_line[2]) / (left_line[1] - left_line[3])) + \
    #                   left_line[0]
    # right_roi_center = (350 - right_line[1]) * ((right_line[0] - right_line[2]) / (right_line[1] - right_line[3])) + \
    #                    right_line[0]
    # print(left_line, right_line, left_roi_center, right_roi_center, 100)
    # left_polygons = np.array([[(left_roi_center - 50, 300), (left_roi_center - 50, 400), (left_roi_center + 50, 400),
    #                            (left_roi_center + 50, 300)]], dtype=int)
    # right_polygons = np.array([[(right_roi_center - 50, 300), (right_roi_center - 50, 400),
    #                             (right_roi_center + 50, 400), (right_roi_center + 50, 300)]], dtype=int)
    # ROI_right_image = display_ROI(frame, ROI_lines(right_polygons))
    # ROI_left_image = display_ROI(frame, ROI_lines(left_polygons))
    # combo_image2 = cv2.addWeighted(combo_image, 1, ROI_right_image, 1, 1)
    # combo_image2 = cv2.addWeighted(combo_image2, 1, ROI_left_image, 1, 1)
    ROI_total = display_ROI_first(frame, ROI_lines(polygons))
    cv2.imshow("sdf", combo_image)
    # combo_image2 = cv2.addWeighted(combo_image2, 1, ROI_total, 1, 1)
    #ROI 이미지 합치기
    theta, l_d = calculate(cross_x, cross_y, (left_location+right_location)/2)

    #세타, 횡방향 오차
    # cv2.imshow("line_image", line_image)
    # cv2.imshow("result", combo_image2)
    # total_result = cv2.hconcat([combo_image, line_image]).copy()
    # total_result = cv2.resize(total_result, (0, 0), fx=0.5, fy=0.5)
    # print(lines.shape[0], 200)
    for i in range(lines.shape[0]):
         cv2.circle(total_result, (int(0.5 * lines[i][0][0]), int(0.5 * lines[i][0][1])), 2, (0, 0, 255), thickness=-1)
         cv2.circle(total_result, (int(0.5 * lines[i][0][2]), int(0.5 * lines[i][0][3])), 2, (0, 0, 255), thickness=-1)
    cv2.imshow("canny", canny_image)
    cv2.imshow("canny_bird_image", canny_bird_image)
    cv2.imshow('total_result', total_result)
    cv2.imshow("combo_image2", combo_image2)


    #랩뷰 전달 부분(socket 이용, 전달값 : send_msg)
    print(theta, l_d, 2)
    msg = str(int(theta))
    msg1 = str(int(l_d))
    # send_msg = msg + ',' + msg1
    # sock.send(send_msg.encode("utf-8"))


    if cv2.waitKey(1) & 0xff == ord('q'):
        break


    # time.sleep(0.04)
cap.release()
cv2.destroyAllWindows()
