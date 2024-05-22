# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
import cv2
import numpy as np
import math
from pathlib import Path
import torch
import socket
from math import pi
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(('localhost', 6340))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode



def auto_canny(image, sigma=0.6):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged

def ROI_lines(ROI):  # polygons(ROI 점들) 수정하면 자동으로 직선 4개 만들기
    return np.array([np.concatenate((ROI[0][0], ROI[0][3]), axis=0), np.concatenate((ROI[0][3], ROI[0][2]), axis=0),
                     np.concatenate((ROI[0][2], ROI[0][1]), axis=0), np.concatenate((ROI[0][1], ROI[0][0]), axis=0)])

def make_line(L_slope, L_intercept, L_y, R_slope, R_intercept, R_y, w, h, y_stack):
    y_top = min(L_y, R_y)
    y_stack2 = np.array([])
    No_Line = 0

    if np.size(y_stack) == 20:
        p1, p3 = np.percentile(y_stack, [40, 60])
        y_stack = np.append(y_stack, y_top)
        y_stack = np.delete(y_stack, [0])
        for i in range(20):
            if p1 <= y_stack[i] <= p3:
                y_stack2 = np.append(y_stack2, y_stack[i])
        y_top = np.average(y_stack2)
    else:
        y_stack = np.append(y_stack, y_top)

    if (L_y != h) and (L_y < R_y):
        lx1 = (y_top - L_intercept) / L_slope
        lx2 = (h - L_intercept) / L_slope
        left_one = np.array([lx1, y_top, lx2, h], dtype=np.int32)
        if (R_y != h):
            rx1 = (y_top - R_intercept) / R_slope
            rx2 = (h - R_intercept) / R_slope
            right_one =np.array([rx1, y_top, rx2, h], dtype=np.int32)
            S_one = np.array([lx1, y_top, rx1, y_top])
        else:
            right_one =np.array([], dtype=np.int32)
            S_one = np.array([lx1, y_top, w-lx1, y_top])
    
    elif (R_y != h) and (L_y > R_y):
        rx1 = (y_top - R_intercept) / R_slope
        rx2 = (h - R_intercept) / R_slope
        right_one =np.array([rx1, y_top, rx2, h], dtype=np.int32)
        if (L_y != h):
            lx1 = (y_top - L_intercept) / L_slope
            lx2 = (h - L_intercept) / L_slope
            left_one =np.array([lx1, y_top, lx2, h], dtype=np.int32)
            S_one = np.array([lx1, y_top, rx1, y_top])
        else:
            left_one =np.array([], dtype=np.int32)
            S_one = np.array([w-rx1, y_top, rx1, y_top])

    else:
        left_one = np.array([], dtype=np.int32)
        right_one = np.array([], dtype=np.int32)
        S_one = np.array([])
        No_Line = 1
    return left_one, right_one, S_one, y_stack, No_Line

def make_one(lines, h):
    line=np.array([[0,0,0,0]])
    if np.size(lines) > 4 :
        slopes = lines[1:,0]
        q1, q3 = np.percentile(slopes, [10, 90])
        for i in range(np.shape(slopes)[0]):
            if (q1 <= slopes[i] <= q3) or (q1 >= slopes[i] >= q3):
                line = np.concatenate([line, np.array([lines[i+1]])])
        y_top = min(line[1:,3])            
        if np.size(line[1:,:]) > 4 :
            line_one = np.average(line[1:,:], axis=0)
            slope = line_one[0]
            intercept = line_one[1]
        else:
            y_top = lines[1][3]
            slope = lines[1][0]
            intercept = lines[1][1]
    else:
        slope = 0
        intercept = 0
        y_top = h
    return slope, intercept, y_top

left_point = np.array([])
right_point = np.array([])
def average_slope_intercept(line_a, w, h, y_stack, left_before, right_before, lines_before):
    left_fit = np.array([[0,0,0,0]])
    right_fit = np.array([[0,0,0,0]])
    No_Line = 0
    try:
        try:
            for line in line_a:
                lines_before = line
                x1, y1, x2, y2 = line.reshape(4)
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                info=np.array([[slope, intercept, x1, y1]])
                if (slope <= -0.8) and (x1 <= 0.5*w):
                    left_fit = np.concatenate([left_fit, info])
                elif (slope > 0.8) and (x1 > 0.5*w):
                    right_fit = np.concatenate([right_fit, info])
        except:
            line_a = lines_before
            x1, y1, x2, y2 = line_a.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            info=np.array([[slope, intercept, x1, y1]])
            if ((slope <= 0) and (x1 <= 0.5*w)):
                left_fit = np.concatenate([left_fit,info])
            elif ((slope > 0) and (x1 > 0.5*w)):
                right_fit = np.concatenate([right_fit,info])
        L_slope, L_intercept, L_y = make_one(left_fit, h)
        R_slope, R_intercept, R_y = make_one(right_fit, h)
        left_one, right_one, S_one, y_stack, No_Line = make_line(L_slope,L_intercept, L_y, R_slope, R_intercept, R_y, w, h, y_stack)
        if len(left_fit) == 1 and len(left_fit) != 1:
            left_one = left_before
        if len(right_fit) == 1 and len(left_fit) != 1:
            right_one = right_before
        if len(left_fit) == 1 and len(left_fit) == 1:
            left_one = left_before
            right_one = right_before
    except:
        left_one = left_before
        right_one = right_before
        S_one = np.array([])
        No_Line = 1
        if len(left_fit) == 1 and len(left_fit) != 1:
                left_one = left_before
        if len(right_fit) == 1 and len(left_fit) != 1:
                right_one = right_before
        if len(left_fit) == 1 and len(left_fit) == 1:
            left_one = left_before
            right_one = right_before
    left_save = left_one
    right_save = right_one
    print(left_one, right_one)
    #     print(left_one, right_one)
    #     # cross_point_x = (right_fit_average[1] - left_fit_average[1]) / (
    #     #             left_fit_average[0] - right_fit_average[0])  # 교차점 x값
    #     # cross_point_y = (left_fit_average[0] * cross_point_x + left_fit_average[1])  # 교차점 y값
    return left_one, right_one, S_one, No_Line, y_stack, left_save, right_save

def replace_pot(left_one, right_one):
    x11 = left_one[0] - 340
    y11 = 480 - left_one[1]
    x12 = left_one[2] - 340
    y12 = 480 - left_one[3]
    x21 = right_one[0] - 340
    y21 = 480 - right_one[1]
    x22 = right_one[2] - 340
    y22 = 480 - right_one[3]
    print(x11, y11, x12, y12, x21, y21, x22, y22)
    return x11, y11, x12, y12, x21, y21, x22, y22

cx = np.array([], dtype = int)
cy = np.array([], dtype = int)
cross_x = np.array([], dtype = int)
cross_y = np.array([], dtype = int)
degree = np.array([], dtype = int)

def control(x11, y11, x12,  y12, x21, y21, x22, y22):
    if x12 == x11 or x22 == x21:
        cx = 10
        cy = 100
        return cx, cy
    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    if m1 == m2:
        print('parallel')
        return None
    cx = np.round((x11 * m1 - y11 - x21* m2 + y21) / (m1 - m2))
    cy = np.round(m1 * (cx - x11) + y11)
    print(cx, cy)
    return cx, cy

def caculate_error(cx, cy):
    if cx < 0:
        horizontal_error = 90 + (np.arctan2(cy, cx) * 180 / pi * (-1))
    elif cx > 0:
        horizontal_error = 90 - np.arctan2(cy, cx) * 180 / pi
    elif cx == 0:
        horizontal_error = 0
    return cx, cy, horizontal_error



# def filt(image):  # filtering 함수
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
#     clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(40, 40))
#     img[:, :, 0] = clahe.apply(img[:, :, 0])
#     img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
#     blur = cv2.GaussianBlur(img, (3, 3), 0)
#     gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 흑백으로 color 조정
#     _, white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#     Lower = np.array([20,0,90])
#     Upper = np.array([30,255,255])
#     yellow = cv2.inRange(blur,Lower,Upper)
#     add = cv2.bitwise_or(white, yellow)
#     canny = auto_canny(add)
#     return canny

def filt(image):  # filtering 함수
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(40, 40))
    # img[:, :, 0] = clahe.apply(img[:, :, 0])
    # img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 흑백으로 color 조정
    # _, white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    Lower = np.array([18,94,140])
    Upper = np.array([48,255,255])
    yellow = cv2.inRange(img,Lower,Upper)
    # add = cv2.bitwise_or(white, yellow)
    canny = auto_canny(yellow)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if np.size(lines) > 0:
        x1, y1, x2, y2 = lines.round(0).astype(int)
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)  # 좌표를 잇는 선 그리기 (255,0,0) = 파란색
    return line_image

def display_ROI(image, lines):  # ROI 빨간선으로 표시하기 위해 따로 생성
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 좌표를 잇는 선 그리기 (0,0,255) = 빨간색
    return line_image

def ROI(image, Roisize):
    mask = np.zeros_like(image)  # image와 같은 shape의 0배열 만듬
    cv2.fillPoly(mask, Roisize, 255)  # 다각형 그리기(image의 mask를 위한것)
    masked_image = cv2.bitwise_and(image, mask)  # image와 mask의 겹치는(and) 이미지 출력
    return masked_image

key = np.zeros(2, dtype = int)
rsize = 0
notraffic = np.zeros((2,2), dtype = int)
middle_x_point = np.zeros(1, dtype = int)
cross_x = 0
degree = 0

def send_message(notraffic, cross_x, cross_y, degree):
    msg1 = str(notraffic[0][0])
    msg2 = str(notraffic[1][0])
    cx_real = str(cross_x)
    cy_real = str(cross_y)
    horizontal_error_real = str(degree)
    print(type(msg1), type(cx_real))
    print(msg1, msg2, cx_real, cy_real, horizontal_error_real)
    msg_s = msg1 + ',' + msg2 + ','+ cx_real + ',' + horizontal_error_real
    # sock.send(msg_s.encode("utf-8"))
    return msg1, msg2, cx_real, horizontal_error_real
def split_line(line):
    try:
        cur = str(line[0])
        cur = cur[7:]
        cur = cur.split('.')
        cur_class = int(cur[0])
        x = str(line[1])
        y = line[2]
        w = line[3]
        h = line[4]
        size = w * h
        return cur_class, y, size  # class와 size 반환
    except Exception as error:
        print('error')
    
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  #
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = True  # source.isnumeric() or source.endswith('.streams') or (is_url and not is_file) # and not이 디폴트
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    left_saved = np.array([1, 1, 1, 1])  # 이전값 저장용 초기값
    right_saved = np.array([1, 1, 1, 1])  # 이전값 저장용 초기값
    lines_saved = np.array([[1, 1, 1, 1],[1, 1, 1, 1]])
    point_saved = np.array([[1, 1, 1, 1],[1, 1, 1, 1]])
    y_stack = np.array([])
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:

            size=np.shape(im0s)
            h=size[1]
            w=size[2]



            # ROI좌표
            polygons = np.array([[(0, 0), (0, 480), (640, 480), (640, 5)]])

            img = im0s[0]

            # 전처리 + ROI로 자르기
            canny_image = filt(img)
            cropped_image1 = ROI(canny_image, polygons)
            cropped_image2 = ROI(canny_image, polygons)
            cropped_image= cv2.bitwise_or(cropped_image1, cropped_image2)
            cv2.imshow('aa',cropped_image)

            # ROI 선 표시
            L_ROI_image = display_ROI(img, ROI_lines(polygons))
            R_ROI_image = display_ROI(img, ROI_lines(polygons))
            ROI_image = cv2.bitwise_or(L_ROI_image, R_ROI_image)
            #
            #if Red:
            #     # 차선 인식


            lines = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 50, maxLineGap=50) #minLineLength=5
            left, right, stop, No_Line, y_stack, left_saved, right_saved = average_slope_intercept(lines, w, h, y_stack, left_saved, right_saved, lines_saved)
            a,b,c,d,e,f,g,h = replace_pot(left, right)
            cross_x, cross_y = control(a,b,c,d,e,f,g,h)
            cross_x, cross_y, degree = caculate_error(cross_x, cross_y)




                # 차선 검출 시 차선 이미지 생성 및 정지선 y 갱신
                # if No_Line == 0:
            S_Line = stop
            L_line_image = display_lines(img, left)
            R_line_image = display_lines(img, right)
            S_line_image = display_lines(img, stop)
            line_image = cv2.bitwise_or(L_line_image, R_line_image)
            line_image = cv2.bitwise_or(line_image, S_line_image)
            combo_image = cv2.bitwise_or(ROI_image, line_image)
            #     else:
            #         combo_image = ROI_image
            #
            #     if np.size(Cars) > 2:
            #         for x, y in Cars[1:,:]:
            #             if (S_Line[0] < x < S_Line[2]) and (y < S_Line[1]):
            #                 Illegal = 1
            #                 print(Illegal)
            # else:
            #     combo_image = ROI_image
            #
            #     Illegal = 0
            #     S_Line = np.array([0,0,0,0])
            #     Red = 0
            #     Arrow = 0
            #     Green = 0
            # Cars = np.array([[0,0]])

            # print(S_Line)

            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'
 
        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    if c == 0:
                        Red = 1
                    if c == 1:
                        Arrow = 1
                    if c == 2:
                        Green = 1

                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # print(xywh)
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        key[0], key[1], rsize = split_line(list(line))
                        if key[0] == 0 or key[0] == 1 or key[0] == 2:
                            if key[0] == 0 or key[0] == 1 or key[0] == 2:
                                if key[1] < notraffic[0][1] - 5:
                                    notraffic[0] = key
                                elif key[1] < notraffic[1][1] - 5:
                                    notraffic[1] = key
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                combo_image2 = cv2.addWeighted(combo_image, 0.8, im0, 1, 1)
                cv2.imshow(str(p), combo_image2)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        send_message(notraffic, cross_x, cross_y, degree)
        notraffic[0][1] = 1000
        notraffic[1][1] = 1000
        notraffic[0][0] = 99
        notraffic[1][0] = 99

        # video = cv2.VideoCapture(1)
        # prev_time = 0
        # FPS = 10
        # while True:
        #     ret, frame = video.read()
        #
        #     current_time = time.time() - prev_time
        #     print(current_time)
        #
        #     if (ret is True) and (current_time > 1. / FPS):
        #         prev_time = time.time()


        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
#http://192.168.1.101:8080/stream

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='C:/Users/EVA/Desktop/yolov5-master/F0929.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default='C:/Users/EVA/Desktop/yolov5-master/stream.txt', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='MS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_false', help='show results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # parser.add_argument("--ip", help="a dummy argument to fool ipython", default="127.0.0.1")
    # parser.add_argument("--stdin", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--control", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--hb", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--Session.key", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--Session.signature_scheme", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--shell", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--transport", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--iopub", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument("--f", help="a dummy argument to fool ipython", default="1")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt




def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
