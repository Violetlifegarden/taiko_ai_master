import numpy as np
import cv2
import torch
import win32gui

taiko_score_region=(1565,0,355,70)
taiko_acc_region=(1745,120,120,40)
taiko_play_region = (0,350,800,190)
def get_width()->int:
    return taiko_play_region[2]//10
def get_height()->int:
    return taiko_play_region[3]//10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""图像截取"""
import mss
def get_screen(region:tuple=(0,350,800,190))->torch.tensor:
        return torch.from_numpy(get_screen_OCR(region)).permute(2,0,1).to(device).float()/255.0
def get_screen_OCR(region:tuple)->np.ndarray:
    params = {
        "left": region[0],  # 左边界 X 坐标
        "top": region[1],  # 上边界 Y 坐标
        "width": region[2],  # 区域宽度
        "height": region[3],  # 区域高度
    }
    with mss.mss() as sct:
        x= cv2.cvtColor(np.array(sct.grab(params)), cv2.COLOR_BGRA2RGB)
        return cv2.resize(x,(get_width(),get_height()),fx=0,fy=0,interpolation=cv2.INTER_AREA)
"""OCR识别"""
import easyocr
def get_scores_and_acc(ocr,wndw,score_region:tuple=(1565,0,1920,70),acc_region:tuple=(1745,120,120,40),):
    score_img = get_screen_OCR(score_region)
    acc_img = get_screen_OCR(acc_region)
    with torch.no_grad():
        score = ocr.readtext(score_img, detail=0)
        acc = ocr.readtext(acc_img, detail=0)
        if score:
            score= int(score[0].replace('o', '0')) if score[0].replace('o', '0').isdigit() else -1
        else:
            score= -1
        if not(score_img.sum() or win32gui.GetWindowText(wndw) == 'osu!'):
            return -1, -1
        if acc:
            acc = int(acc[0].replace('o', '0').replace('.', '0')) / 1000 if acc[0].replace('o', '0').replace('.','0').isdigit() else -1
        else:
            acc = -1
        return score,acc

"""模拟鼠标点击"""

"""经验池（要么就用原来的）"""
class ReplayMemory2(object):
    def __init__(self, capacity:int):
        """"""
        self.capacity = capacity
        self.memory = torch.empty((capacity, 4, 3, 19, 80)).to(device)
        self.position = 0

    def push(self, *args):
        self.memory[self.position][0] = args[0]
        self.memory[self.position][1][0, 0, 0] = args[1].squeeze(0)
        self.memory[self.position][2][0, 0, 0] = args[2][-1]
        self.memory[self.position][3] = args[3]
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sampling = torch.randint_like(torch.zeros(batch_size), 0, self.capacity, dtype=torch.long).to(device)
        batch = [self.memory[i] for i in sampling]
        s = torch.stack([a[0] for a in batch])
        a = torch.stack([a[1][0, 0, 0].type(torch.long) for a in batch])
        r = torch.stack([a[2][0, 0, 0] for a in batch])
        s1 = torch.stack([a[3] for a in batch])

        return s, a, r, s1

    def __len__(self):
        return len(self.memory)
