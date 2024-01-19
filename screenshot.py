from mss import mss
import numpy
import cv2
ScreenX=1920
ScreenY=1080
window_size=(
    int(ScreenX/2-320),#960
    int(ScreenY/2-320),#480
    int(ScreenX/2+320),#1600
    int(ScreenY/2+320)#1120
)
Screenshot_value=mss()

def screenshot():
    img=Screenshot_value.grab(window_size)
    img=numpy.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
    return img

# while True:
#     cv2.imshow('a',screenshot())
#     cv2.waitKey(0)