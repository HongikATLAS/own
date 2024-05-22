import cv2
import numpy as np

def detect_lines(image):
    # 이미지 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 120, 300)

    # ROI 설정
    height, width = image.shape[:2]
    roi_vertices = [(0, height), (width/2, height/2), (width, height)]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 허프 변환
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=10, minLineLength=100, maxLineGap=5)

    # 선분 합치기 및 시각화
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    # 원본 이미지와 검출된 라인 이미지 합성
    result = cv2.addWeighted(image, 1, line_image, 0.8, 0)

    return result

# 이미지 읽어오기
image = cv2.imread('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/lane2.jpg')

# 라인 검출 함수 호출
result = detect_lines(image)

# 결과 이미지 출력
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과값이 개 븅신처럼 나온다 ㄹㅇ / 버러지 코드 ㄹㅇ
