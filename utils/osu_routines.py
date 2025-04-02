import subprocess
import time

import psutil
import pyautogui
import pyclick
import win32api
import win32con
import win32gui

KEY_DICT = {' ': 0x20,
            '0': 0x30,
            '1': 0x31,
            '2': 0x32,
            '3': 0x33,
            '4': 0x34,
            '5': 0x35,
            '6': 0x36,
            '7': 0x37,
            '8': 0x38,
            '9': 0x39,
            'a': 0x41,
            'b': 0x42,
            'c': 0x43,
            'd': 0x44,
            'e': 0x45,
            'f': 0x46,
            'g': 0x47,
            'h': 0x48,
            'i': 0x49,
            'j': 0x4A,
            'k': 0x4B,
            'l': 0x4C,
            'm': 0x4D,
            'n': 0x4E,
            'o': 0x4F,
            'p': 0x50,
            'q': 0x51,
            'r': 0x52,
            's': 0x53,
            't': 0x54,
            'u': 0x55,
            'v': 0x56,
            'w': 0x57,
            'x': 0x58,
            'y': 0x59,
            'z': 0x5A, 'backspace': 0x08}


def start_osu():
    try:
        for proc in psutil.process_iter():
            if proc.name() == 'osu!.exe':
                proc.kill()
        time.sleep(0.5)
        process = subprocess.Popen(r'C:/Users/Rain/Desktop/osu!/osu!.exe')
        p = psutil.Process(process.pid)
        time.sleep(5)
        osu_window = win32gui.FindWindow(None, "osu!")
        width, height = win32gui.GetWindowRect(osu_window)[2] - win32gui.GetWindowRect(osu_window)[0], \
                        win32gui.GetWindowRect(osu_window)[3] - win32gui.GetWindowRect(osu_window)[1]
        win32gui.MoveWindow(osu_window, -3, 0, width, height, False)
    except Exception as e:
        print(e)
        p = None

    return p, osu_window


def stop_osu(process):
    if process is None:
        return
    try:
        process.kill()
    except Exception as e:
        print(e)
    return


def move_to_songs(star=1):
    hc = pyclick.HumanClicker()
    time.sleep(0.5)
    hc.move((969, 578), 0.1)
    hc.click()
    hc.move((1266, 335), 0.5)
    hc.click()
    time.sleep(0.2)
    hc.click()
    time.sleep(0.5)
    launch_random_beatmap()
    if star is not None:
        hc.move((1750, 110 + (star-1) * 60), 1)
        time.sleep(0.5)
        hc.click()
    time.sleep(0.4)
    hc.move((450, 320), 0.8)
    del hc
    return


def launch_random_beatmap():######
    hc = pyclick.HumanClicker()
    hc.move((607,991), 0.25)
    time.sleep(0.1)
    hc.click()
    time.sleep(3.3)
    return


def select_beatmap(search_name):
    hc = pyclick.HumanClicker()
    pyautogui.mouseUp(button='left')
    pyautogui.mouseUp(button='right')
    time.sleep(0.5)
    for letter in search_name:
        win32api.keybd_event(KEY_DICT[letter], 0, 0, 0)
        time.sleep(0.04)
        win32api.keybd_event(KEY_DICT[letter], 0, win32con.KEYEVENTF_KEYUP, 0)
    hc.move((1750, 689), 2.5)
    time.sleep(0.8)
    hc.click()
    return


def launch_selected_beatmap():#已完成
    hc = pyclick.HumanClicker()
    time.sleep(0.2)
    hc.move((1820,972), 0.6)
    time.sleep(0.2)
    hc.click()
    return

def return_to_beatmap():#######
    hc = pyclick.HumanClicker()
    time.sleep(8)
    hc.move((1314,1082), 0.8)
    time.sleep(0.2)
    hc.click()
    time.sleep(0.4)
    hc.click()
    time.sleep(0.4)
    return
