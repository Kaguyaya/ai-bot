# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
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
import os
import platform
import sys
import threading
import pyautogui
from sendinput import *
import time
from pathlib import Path
import numpy as np
import math
import pynput.mouse
from pynput.mouse import Listener
import torch
from utils.augmentations import letterbox
import pydirectinput
import ctypes
import os
import pynput
import winsound
from pynput import keyboard

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from screenshot import screenshot


is_F10_pressed = False
def mouse_click(x, y, button, pressed):
    global is_x2_pressed
    # print(x,y,button,pressed)
    if pressed and button == pynput.mouse.Button.x2:
        print('开始瞄准')
        is_x2_pressed = True
    elif not pressed and button == pynput.mouse.Button.x2:
        print('瞄准关闭')
        is_x2_pressed = False


def mouse_listener():
    with keyboard.Listener(on_press=key_press, on_release=key_release) as listener:
        listener.join()

def key_press(key):
        global is_F10_pressed
        if key == keyboard.Key.f9:
            is_F10_pressed = True



def key_release(key):
        global is_F10_pressed
        if key == keyboard.Key.f9:
            is_F10_pressed = False

try:
    root = os.path.abspath(os.path.dirname(__file__))
    driver = ctypes.CDLL(f'{root}/logitech.driver.dll')
    ok = driver.device_open() == 1  # 该驱动每个进程可打开一个实例
    if not ok:
        print('Error, GHUB or LGS driver not found')
except FileNotFoundError:
    print(f'Error, DLL file not found')


class Logitech:
    class mouse:

        """
        code: 1:左键, 2:中键, 3:右键
        """

        @staticmethod
        def press(code):
            if not ok:
                return
            driver.mouse_down(code)

        @staticmethod
        def release(code):
            if not ok:
                return
            driver.mouse_up(code)

        @staticmethod
        def click(code):
            if not ok:
                return
            driver.mouse_down(code)
            driver.mouse_up(code)

        @staticmethod
        def scroll(a):
            """
            a:没搞明白
            """
            if not ok:
                return
            driver.scroll(a)

        @staticmethod
        def move(x, y):
            """
            相对移动, 绝对移动需配合 pywin32 的 win32gui 中的 GetCursorPos 计算位置
            pip install pywin32 -i https://pypi.tuna.tsinghua.edu.cn/simple
            x: 水平移动的方向和距离, 正数向右, 负数向左
            y: 垂直移动的方向和距离
            """
            if not ok:
                return
            if x == 0 and y == 0:
                return
            driver.moveR(x, y, True)

    class keyboard:

        """
        键盘按键函数中，传入的参数采用的是键盘按键对应的键码
        code: 'a'-'z':A键-Z键, '0'-'9':0-9, 其他的没猜出来
        """

        @staticmethod
        def press(code):

            if not ok:
                return
            driver.key_down(code)

        @staticmethod
        def release(code):
            if not ok:
                return
            driver.key_up(code)

        @staticmethod
        def click(code):
            if not ok:
                return
            driver.key_down(code)
            driver.key_up(code)





@smart_inference_mode()
def run():
    # Load model 加载模型
    device = select_device('cuda:0')
    model = DetectMultiBackend('./weight/yolov5n.pt', device=device, dnn=False, data=False, fp16=True)
    # 读取图片
    while True:
        if is_F10_pressed:
            im = screenshot()
            im0 = im
            im = letterbox(im, (640, 640), stride=32, auto=True)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
                # 推理

                pred = model(im, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=0, max_det=1000)

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # Process predictions
            for i, det in enumerate(pred):  # per image
                annotator = Annotator(im0, line_width=1)
                if len(det):
                    distance_list = []
                    target_list = []
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):  # 处理推理出来每个目标的信息
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        line = cls, *xywh, conf

                        X = xywh[0] - 320
                        Y = xywh[1] - 320
                        distance = math.sqrt(X ** 2 + Y ** 2)
                        xywh.append(distance)
                        annotator.box_label(xyxy, label=f'[{int(cls)}Distance:{round(distance, 2)}]', color=(34, 139, 34),
                                            txt_color=(0, 191, 255))
                        distance_list.append(distance)
                        target_list.append(xywh)

                    # print('距离',distance_list)
                    # print('目标信息',target_list)
                    # print(distance_list.index(min(distance_list)))
                    target_info = target_list[distance_list.index(min(distance_list))]
                    # mouse_xy(int(target_info[0]-320),int(target_info[1]-320))

                    print(target_info)
                    print(target_info[0]-320)
                    print(target_info[1]-320)
                    # print(pydirectinput.position())

                    Logitech.mouse.move(int(target_info[0]-320), int(target_info[1]-320))
                    # pydirectinput.moveRel(int(target_info[0]-320),int(target_info[1]-320),duration=0)
                im0 = annotator.result()
                cv2.imshow('window', im0)
                cv2.waitKey(1)
                time.sleep(0.01)


if __name__ == "__main__":
    threading.Thread(target=mouse_listener).start()
    # 监听键盘键入





    run()
