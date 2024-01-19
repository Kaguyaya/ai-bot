import pynput.mouse
from pynput.mouse import Listener

is_x2_pressed=False
def mouse_click(x,y,button,pressed):
    global is_x2_pressed
    print(x,y,button,pressed)
    if pressed and button==pynput.mouse.Button.x2:
        print('开始瞄准')
        is_x2_pressed=True
    elif not pressed and button==pynput.mouse.Button.x2:
        print('瞄准')
        is_x2_pressed=False

with Listener(on_click=mouse_click) as listener:
    listener.join()