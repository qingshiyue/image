import os
import time

"""
    适用于windows操作环境
"""

def adb_start_activity(activity):
    """
    打开app指定activity
    :param activity: activity 名称string类型
    :return:
    """
    adb_shell = 'adb shell am start -n '
    # 点击坐标位置
    os.system( adb_shell + activity )
    print(adb_shell + activity)
    time.sleep(0.1)

def adb_tap(size):
    '''
    坐标点击
    :param size: [x,y]
    :return:
    '''
    adb_shell = 'adb shell input tap '
    # 点击坐标位置
    os.system( adb_shell + str(size[0]) + ' ' + str(size[1]) )
    time.sleep(0.1)


def adb_input_text(text):
    '''
    向输入框输入文字
    :param text: 'str' 字符串，utf-8
    :return:
    '''
    adb_shell = 'adb shell am broadcast -a ADB_INPUT_TEXT --es msg '
    # 虚拟键盘安装 adb install ADBKeyboard.apk
    # 调出adbkeyboard虚拟键盘
    os.system('adb shell ime set com.android.adbkeyboard/.AdbIME')
    # 虚拟键盘输入
    os.system(adb_shell + '\'' + text + '\'')
    time.sleep(0.1)


def adb_screencap(filename):
    '''
    屏幕截图保存
    :param filename: 截图保存路径（电脑端 仅支持png）
    :return:
    '''
    adb_shell = 'adb shell screencap -p /sdcard/adb_screencap.png'
    os.system(adb_shell)
    adb_shell = 'adb pull /sdcard/adb_screencap.png '+ filename
    os.system(adb_shell)
    time.sleep(0.1)

def adb_swipe(size, time):
    '''
    屏幕滑动
    :param size: [(20,20),(20,100)] 起始点
    :param time: 100 滑动时间单位：ms
    :return:
    '''
    adb_shell = 'adb shell input swipe '
    for s in size:
        adb_shell += str(s[0]) + ' ' + str(s[1]) + ' '
    adb_shell += str(int(time))
    os.system(adb_shell)
    time.sleep(0.1)