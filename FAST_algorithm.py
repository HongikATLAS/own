import numpy as np
import cv2

def fast():
    img = cv2.imread("C:/Users/kchw3_r2l9a77/OneDrive/Desktop/lunar.jpg")
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("imgray", imgray)
    img2, img3 = None, None

    fast = cv2.FastFeatureDetector_create(150)
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, img2, (255, 0,0))
    cv2.imshow("img2", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

fast()
