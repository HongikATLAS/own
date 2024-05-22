import cv2
vidcap = cv2.VideoCapture('C:/Users/82105/Downloads/c.mp4')
count = 0
while (vidcap.isOpened()):
  success,image = vidcap.read()
  if success and count % 24 == 0:
    cv2.imwrite("C:/Users/82105/Downloads/c/%07d.jpg" % count, image)     # save frame as JPEG file
    print('Read a new frame: ', success)
    print(count)
  elif success == False:
    break
  count += 1
vidcap.release()
print("finish! convert video to frame")