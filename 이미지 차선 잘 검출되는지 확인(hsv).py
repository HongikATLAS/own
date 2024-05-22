import numpy as np
import cv2
from time import sleep

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
    return np.array([left_line, right_line]), left_save, right_save, cross_point_x, cross_point_y, lines_before, left_line[0],  right_line[0], left_line, right_line

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    try:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # 좌표를 잇는 선 그리기 (255,0,0) = 파란색
    except Exception as error:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (0, 480), (640, 480), (255, 0, 0), 10)  # 값이 이상한 경우가 있어서 예외처리
    return line_image




def main():
    image = cv2.imread('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/lane2.jpg') # 파일 읽어들이기

    left_saved = []  # 이전값 저장용 초기값
    right_saved = []  # 이전값 저장용 초기값
    lines_saved = [[]]

    # HSV로 색추출
    hsvLower = np.array([0, 0, 168])    # 추출할 색의 하한
    hsvUpper = np.array([172, 50, 255])    # 추출할 색의 상한
    hsvResult = hsvExtraction(image, hsvLower, hsvUpper)
    print(hsvResult, hsvResult.shape)

    # cv2.imshow('HSV_test1', hsvResult)
    blur = cv2.GaussianBlur(hsvResult, (5, 5), 0)
    # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
    canny = cv2.Canny(blur, 50, 300)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 20, np.array([]), minLineLength=10, maxLineGap=2)
    # minlinelength가 너무 낮으면 너무 짧은 선이 검출된다 / max_line_gap보다 점 사이의 거리가 크면 나와 다른 선
    print(lines.shape[0]) #  이걸로 선 개수 확인
    for i in range(lines.shape[0]):
        cv2.circle(image, (int(lines[i][0][0]), int(lines[i][0][1])), 2, (0, 0, 255), thickness=-1)
        cv2.circle(image, (int(lines[i][0][2]), int(lines[i][0][3])), 2, (0, 0, 255), thickness=-1)
    # cv2.imshow("hsv", hsvResult) # 차선만 나오는지 확인A
    cv2.imshow("canny", canny) # 모서리가 잘 검출되는지 확인
    cv2.imshow("line", image) # 선 잘 나오는지 확인


    while True:
        if cv2.waitKey() == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()