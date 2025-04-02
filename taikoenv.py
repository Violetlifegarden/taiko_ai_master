import time
from typing import Optional

import easyocr
import gym
import keyboard
import pyautogui
import torch
from gym import spaces
from gym.core import ActType

import taiko_utils
from taiko_utils import taiko_acc_region,taiko_score_region,taiko_play_region
import utils.osu_routines

#from utils.screen import TAIKO_REGION
"""根据自己屏幕分辨率来确定，默认为1920*1080的情况"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TaikoEnv(gym.Env):
    def __init__(self):
        """
        last_action:ActType
        stack_size:int
        history:torch.Tensor
        """
        super(TaikoEnv,self).__init__()
        self.process,self.wndw = utils.osu_routines.start_osu()
        self.stack_size = 3
        self.steps = 0
        self.episode_counter = 0
        self.action_space = spaces.Discrete(3)

        self.last_action = None
        self.previous_score = None
        self.previous_acc = None
        self.history = None

        #self.key_dict = {'c': 0x43, 'v': 0x56}
        self.ocr = easyocr.Reader(['ch_sim'])


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) :
        self.episode_counter +=1
        self.steps = 0
        self.previous_score = 0
        self.previous_acc = 0
        tmp = self.get_obs()
        self.history = torch.zeros((self.stack_size, tmp.shape[1], tmp.shape[2])).to(device)
        self.history = tmp
        return self.history



    def step(self, action:ActType) :
        """
        1.执行动作
        2.使用OCR识别分数 问题：取多少帧后的图像作为识别对象？
        """
        self.fake_action(action)
        self.steps +=1
        self.last_action = action
        
        score,acc = taiko_utils.get_scores_and_acc(ocr=self.ocr,score_region=taiko_score_region,acc_region=taiko_acc_region,wndw=self.wndw)
        #self.history[:-1] = self.history[1:]
        print(f"time:{time.time()}\nscore:{score}\nacc:{acc}\n\n")
        self.history = self.get_obs()

        done = (score==acc==-1)
        if score - self.previous_score > 2000.0:
            rew = torch.tensor([0.0], device=device)
        else:
            rew = self.get_reward(score,acc)
        if self.steps < 15:
            done = False

        self.previous_score = max(self.previous_score, score)
        self.previous_acc = acc
        return self.history, rew, done

    @staticmethod
    def get_obs():
        """应该是获取taiko阅读屏幕的区域，等下需要看下略过帧应该在哪写"""
        return taiko_utils.get_screen(taiko_play_region)


    def get_reward(self, score,acc)->torch.tensor:
        """if score !=-1 and acc !=-1:"""
        return torch.tensor([max(score - self.previous_score, 0)/1000.0+max((acc-self.previous_acc),0)], device=device)
        """else:
            if score == -1:
                return torch.tensor([0],device=device) if acc<self.previous_acc else torch.tensor([acc-self.previous_acc+1],device=device),
            elif acc ==-1:
                return torch.tensor([max(score - self.previous_score, 0)/1000],device=device)
            else:
                return torch.tensor([0],device=device)"""

    @staticmethod
    def fake_action(action)->None:
        if action.item() == 1:
            pyautogui.click(button='left')
        elif action.item() == 2:
            pyautogui.click(button='right')
    def start_game(self):
        """是从选歌界面开始，所以说需要随机选歌加一个点击osu"""
        pass

