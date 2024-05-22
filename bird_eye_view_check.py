import cv2
import numpy as np
from matplotlib import pyplot as plt

FRAME_WIDTH = 640
FRAME_HEIGTH = 480
# cv2.CAP_DSHOW+0
cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)#'C:/Users/kchw3_r2l9a77/Downloads/lane.mp4' #'C:/Users/kchw3_r2l9a77/OneDrive/Desktop/test.mp4'
print('width:', cap.get(3), ' height:', cap.get(4))

if cap.isOpened():  # 캡처 객체 초기화 확인
    while True:
        ret, img = cap.read()  # 다음 프레임 읽기
        if ret:  # 프레임 읽기 정상
            # cv2.imshow('VIDEO', img)
            pts1 = np.float32([[200, 0], [50, 360], [590, 360], [440, 0]])
            # 좌표의 이동점
            pts2 = np.float32([[0, 0], [0, 480], [640, 480], [640, 0]])
            object_plot = np.zeros(3)
            object_plot = 300, 200, 1
            object_plot = np.transpose(object_plot)
            print(object_plot)

            # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
            M = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
            dst = cv2.warpPerspective(img, M, (640, 480))
            dot = M@object_plot
            print(dot)
            dot_x = int(dot[0]/dot[2])
            dot_y = int(dot[1]/dot[2])

            print(dot_x, dot_y)
            # 여기서 640, 480은 내가 원하는 변환된 이미지 사이즈 크기
            # dst_min_or = dst.min()
            # dst_max_or = dst.max()
            # dst_img_unit = ((dst - dst_min_or) / (dst_max_or - dst_min_or) * 255.).astype(np.uint8)  # change type
            # img_min_or = img.min()
            # img_max_or = img.max()
            # img_unit = ((img - img_min_or) / (img_max_or - img_min_or) * 255.).astype(np.uint8)
            cv2.circle(dst, (dot_x, dot_y), 10, (0, 0, 255), thickness=-1)
            hcon = cv2.hconcat([img, dst]).copy()
            hcon = cv2.resize(hcon, (0, 0), fx=0.5, fy=0.5)
            # cv2.imshow("circle", dst)
            cv2.imshow('bird-eye-view', hcon)
            # cv2.imshow("df", img)
            # cv2.imshow("sdfsf", dst)
            points = np.array([[25, 0, 0, 1]], dtype=np.float32)



            # x = 100
            # y = 1006
            # w = 150
            # h = 150
            # roi = dst[y:y+h,x:x+h]
            # cv2.rectangle(roi, (0,0), (h-1, w-1), (0,255,0))
            # cv2.imshow('img', dst)

            # cv2.imshow('camera', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
else:
    print(2)

