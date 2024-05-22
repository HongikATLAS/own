import cv2
import numpy as np
#
# def detect_lines(image):
#     # 이미지 전처리
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#
#     # ROI 설정
#     height, width = image.shape[:2]
#     roi_vertices = [(0, height), (width/2, height/2), (width, height)]
#     mask = np.zeros_like(edges)
#     cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
#     masked_edges = cv2.bitwise_and(edges, mask)
#
#     # 허프 변환
#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
#
#     # 선분 합치기 및 시각화
#     line_image = np.zeros_like(image)
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
#
#     # 원본 이미지와 검출된 라인 이미지 합성
#     result = cv2.addWeighted(image, 1, line_image, 0.8, 0)
#
#     return result
#
# # 이미지 읽어오기
# image = cv2.imread('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/test..jpg')
#
# # 라인 검출 함수 호출
# result = detect_lines(image)
#
# # 결과 이미지 출력
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

src = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# 에지 검출
edge = cv2.canny(src, 50, 150)

# 직선 성분 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180., 160, minLineLength=50, maxLineGap=5)

# 컬러 영상으로 변경 (영상에 빨간 직선을 그리기 위해)
det = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

if lines is not None:  # 라인 정보를 받았으면
    for i in range(lines.shape[0]):
        pt1 = (lines[i][0][0], lines[i][0][1])  # 시작점 좌표 x,y
        pt2 = (lines[i][0][2], lines[i][0][3])  # 끝점 좌표, 가운데는 무조건 0
        cv2.line(dst, pt1m
        pt2m(0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('src', src)
        cv2.imshow('dst', dst)
        cv2.waitKey()
        cv2.destroyAllWIndows()