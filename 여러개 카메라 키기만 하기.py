import cv2
import time

def filt(image):  # filtering 함수
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 흑백으로 color 조정
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 평균값 필터 블러링 -> 모든 사진의 가중치를 평균값으로 조정(즉 멀리있는 픽셀의 퀄리티가 낮아지는 것을 보완)
    canny = cv2.Canny(blur, 250, 300)
    # edge만 추출, 임계값이 클수록 검출이 적게되고 작을수록 검출이 많이됨(실험적으로 픽셀을 잘라내야 함)
    return canny

FRAME_WIDTH = 640
FRAME_HEIGTH = 480

cap = cv2.VideoCapture(cv2.CAP_DSHOW+2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGTH)

while True:
    # curr_time = time.time()
    ret, frame = cap.read() #프레임 단위로 읽음
    cv2.imshow("frame", frame)
    canny = filt(frame)
    cv2.imshow("canny", canny)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    # time.sleep(0.04)
cap.release()
cv2.destroyAllWindows()
