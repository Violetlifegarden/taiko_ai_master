import pyautogui
import time
import osu_routines


def mouse_locator():
    while True:
        time.sleep(0.5)
        x, y = pyautogui.position()
        print('(' + str(x) + ', ' + str(y) + ')')

if __name__ == '__main__':
    osu_routines.start_osu()
    mouse_locator()

