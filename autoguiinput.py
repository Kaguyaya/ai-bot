from pynput import keyboard

# 监听键盘键入
def key_press(key):
    if key == keyboard.Key.f10:
        return True

def key_release(key):
    if key == keyboard.Key.f10:
        return False


with keyboard.Listener(on_press=key_press,on_release=key_release) as listener:
    listener.join()