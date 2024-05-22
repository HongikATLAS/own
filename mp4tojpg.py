import cv2
vidcap = cv2.VideoCapture('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/threelight.mp4')
success,image = vidcap.read()
count = 0
while success:
  if count % 8 == 0:
    cv2.imwrite("C:/Users/kchw3_r2l9a77/OneDrive/Desktop/threelight/%07d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  print(count)
  count += 1


print("finish! convert video to frame")