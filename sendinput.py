from ctypes import windll,c_long,c_ulong,Structure,Union,c_int,POINTER,sizeof

LONG=c_long
DWORD=c_ulong
ULONG_PTR=POINTER(DWORD)

class MOUSEINPUT(Structure):
    _fields_ = (('dx',LONG),
                ('dy',LONG),
                ('mouseData',DWORD),
                ('dwFlags',DWORD),
                ('time',DWORD),
                ('dwExtraInfo',ULONG_PTR)
                )

class _INPUTunion(Union):
    _fields_ = (('mi',MOUSEINPUT),('mi',MOUSEINPUT))

class INPUT(Structure):
    _fields_ = (('type',DWORD),
                ('union',_INPUTunion)
                )

def SendInput(*inputs):
    nInputs=len(inputs)
    LPINPUT=INPUT*nInputs
    pInputs=LPINPUT(*inputs)
    cbsize=c_int(sizeof(INPUT))
    return windll.user32.SendInput(nInputs,pInputs,cbsize)

def Input(structure):
    return INPUT(0,_INPUTunion(mI=structure))

def MouseInput(flags,x,y,data):
    return MOUSEINPUT(x,y,data,flags,0,None)

def Mouse(flags,x=0,y=0,data=0):
    return Input(MouseInput(flags,x,y,data))

def mouse_xy(x,y):
    return SendInput(Mouse(0x0001,x,y))

def mouse_down(key=1):
    if key==1:
        return SendInput(Mouse(0x0002))
    elif key==2:
        return SendInput(Mouse(0x0008))

def mouse_up(key=1):
    if key==1:
        return SendInput(Mouse(0x0004))
    elif key==2:
        return SendInput(Mouse(0x0010))