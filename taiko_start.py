from threading import Thread

import pyautogui
import torch

import utils.osu_routines
from taiko_trainer import TaikoQTrainer
from taikoenv import TaikoEnv

torch.set_printoptions(sci_mode=False)

pyautogui.MINIMUM_DURATION = 0.0
pyautogui.MINIMUM_SLEEP = 0.0
pyautogui.PAUSE = 0.0

BATCH_SIZE = 32
LEARNING_RATE = 0.00005
GAMMA = 0.999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def TaikoTrain(lr=0.00008, batch_size=32, episode_nb=50, min_experience=10000,
               root_dir='./weights',):
    env = TaikoEnv()
    trainer = TaikoQTrainer(env, batch_size=batch_size, lr=lr, gamma=GAMMA, root_dir=root_dir, min_experience=min_experience)
    utils.osu_routines.move_to_songs()
    utils.osu_routines.launch_selected_beatmap()
    for episode in range(episode_nb):
        state = env.reset()
        env.start_game()
        while True:
            action = trainer.select_action(state)
            new_state, reward, done = env.step(action)
            if done:
                #env.next_game()
                break
            Thread(target=trainer.memory.push,
                   args=(state, action, reward, new_state)).start()
            state = new_state

        """如果使用cpu推理那么速度显然很慢
        但我们要求每玩完一轮要重新更新模型后才能进入下一轮
        之后会考虑改异步
        """
        for i in range(1000):
            trainer.optimize()
        #print("-----------------------------------------------\n"*10)
        #trainer.checkpointer.save()
        #trainer.update_target()
    trainer.stop()







