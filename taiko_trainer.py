import _pickle as cPickle
import bz2
import math
import random

import torch
import torch.nn.functional as F

import utils.checkpoint
import utils.osu_routines
from taikoenv import TaikoEnv
from taiko_utils import ReplayMemory2,get_width,get_height
from taikomodel import TaikoQFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TaikoQTrainer:
    def __init__(self,env:TaikoEnv,batch_size=32, lr=0.0001, gamma=0.999, eps=1.5e-4,root_dir='./weights',min_experience=3000,):
        self.env =env
        self.checkpointer = utils.checkpoint.Checkpointer(root_dir)

        load_memory, load_optimizer, load_network, load_steps = self.checkpointer.load()

        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma

        self.q_network = TaikoQFunction(width=get_width(),height=get_height()).to(device)
        self.target_q_network = TaikoQFunction(width=get_width(),height=get_height()).to(device)
        if load_network is not None:
            self.q_network.load_state_dict(torch.load(load_network))
            self.target_q_network.load_state_dict(torch.load(load_network))
        if load_memory is None:
            self.memory = ReplayMemory2(min_experience)
        else:
            with bz2.open(load_memory, 'rb') as f:
                self.memory = cPickle.load(f)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr, eps=eps)
        if load_optimizer is not None:
            self.optimizer.load_state_dict(torch.load(load_optimizer))
        if load_steps is not None:
            with bz2.open(load_steps, 'rb') as f:
                self.steps_done = cPickle.load(f)
        else:
            self.steps_done = 0

        self.min_experience = min_experience

        self.start_epsilon = 0.99
        self.end_epsilon = 0.05



    def optimize(self):
        if len(self.memory) < self.min_experience:
            return
        s0, a, r, s1 = self.memory.sample(batch_size=self.batch_size)
        y = self.q_network(s0)
        state_action_values = torch.stack([y[i, a[i]] for i in range(self.batch_size)])  # Get estimated Q(s1,a1)
        next_state_values = self.target_q_network(s1).detach().max(1)[0]
        expected_state_action_values = r.squeeze() + self.gamma * next_state_values
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        #print(f"time={time.time()}")
        #print(f"loss={loss}\n")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return

    def select_action(self, state):
        self.steps_done +=1
        with torch.no_grad():
            return self.q_network(state).max(1)[1] if (random.random()<self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(-1.0 * self.steps_done / 10000)) else self.random_action()

    @staticmethod
    def random_action():
        return torch.tensor([random.randrange(3)], device=device, dtype=torch.int)


    def update_target(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        return
    def stop(self):
        self.checkpointer.save(self.memory, self.target_q_network, self.optimizer, self.steps_done)
        utils.osu_routines.stop_osu(self.env.process)
        return