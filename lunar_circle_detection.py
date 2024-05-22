import numpy as np
import cv2

def filt(image):  # filtering 함수
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 흑백으로 color 조정
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
    canny = cv2.Canny(blur, 300, 500)
    # edge만 추출, 임계값이 클수록 검출이 적게되고 작을수록 검출이 많이됨(실험적으로 픽셀을 잘라내야 함)
    return canny

image = cv2.imread("C:/Users/kchw3_r2l9a77/OneDrive/Desktop/lunar.jpg")

    #     continue
frame = np.array(image)
canny_image = filt(frame)
frame = cv2.resize(frame, dsize=(500,int(895/2)), interpolation=cv2.INTER_LINEAR)
canny_image = cv2.resize(canny_image, dsize=(500,int(895/2)), interpolation=cv2.INTER_LINEAR)
cv2.imshow("frame", frame)
cv2.imshow("canny", canny_image)
# print(canny_image, canny_image.shape)
canny_image = np.float32(canny_image)
print(canny_image, type(canny_image))
a = canny_image.shape[0]
b = canny_image.shape[1]
print(a, type(a))
for i in range(a):
    for j in range(b):
        print(j, end=' ')

cv2.waitKey(0)
