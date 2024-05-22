import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/img.jpg')

p1 = [230, 900]  # 좌상 1024  512
p2 = [30, 1250] # 좌하
p3 = [836, 1250] # 우하
p4 = [616, 900]  # 우상

cv2.circle(image, (p1[0], p1[1]), 15, (0, 0, 255), thickness=-1)
cv2.circle(image, (p2[0], p2[1]), 15, (0, 0, 255), thickness=-1)
cv2.circle(image, (p3[0], p3[1]), 15, (0, 0, 255), thickness=-1)
cv2.circle(image, (p4[0], p4[1]), 15, (0, 0, 255), thickness=-1)

# corners_point_arr는 변환 이전 이미지 좌표 4개
corner_points_arr = np.float32([p1, p2, p3, p4])
width = image.shape[1]
height = image.shape[0]
print(width, height)

#height가 폭(x)이고 width가 높이(y)이다.
image_p1 = [0, 0]
image_p2 = [0, height]
image_p3 = [width, height]
image_p4 = [width, 0]

image_params = np.float32([image_p1, image_p2, image_p3, image_p4])


mat = cv2.getPerspectiveTransform(corner_points_arr, image_params)
# mat = 변환행렬(3*3 행렬) 반
image_transformed = cv2.warpPerspective(image, mat, (width, height))


hcon = cv2.hconcat([image, image_transformed])
hcon = cv2.resize(hcon, (0, 0), fx=0.5, fy=0.5) # fx와 fy가 1이면 원본 비율 2개를 붙인다.
cv2.imshow('bird-eye-view', hcon)
cv2.waitKey(0)
cv2.destroyAllWindows()