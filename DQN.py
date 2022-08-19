from collections import deque
import random
from random import sample

import numpy as np
import gym
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cpu")

class DQNNet(nn.Module):
    """
    Defines a single DQN Network for the OpenAI Lunar Lander
    environment
    """
    def __init__(self, hidden_layers, layer_size):
        assert hidden_layers==2 or hidden_layers==3
        self.hidden_layers = hidden_layers

        super(DQNNet, self).__init__()
        self.layer1 = nn.Linear(8, layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)
        if self.hidden_layers==3:
            self.layer3 = nn.Linear(layer_size, layer_size)
        self.layer4 = nn.Linear(layer_size, 4)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        if self.hidden_layers==3:
            x = F.relu(self.layer3(x))
        return self.layer4(x)

class DQNAgent():
    def __init__(self, hidden_layers, layer_size, gamma,
              replay_size, target_update, start_e, beta,
              e_min, alpha, batch_size, seed=1):

        # Initialize networks with same parameters
        self.train_net = DQNNet(hidden_layers, layer_size)
        self.target_net = DQNNet(hidden_layers, layer_size)
        self.target_net.eval()
        self.target_net.load_state_dict(self.train_net.state_dict())

        # Initialize attributes and hyper-parameters
        self.env = gym.make('LunarLander-v2')
        self.gamma = gamma
        self.beta = beta
        self.replay_size = replay_size
        self.replay = deque(maxlen = replay_size)
        self.target_update = target_update
        self.batch_size = batch_size

        self.epsilon = start_e
        self.beta = beta
        self.e_min = e_min

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.train_net.parameters(), lr=alpha)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed(seed)

    def sample_from_exp_replay(self):
        exp_replay = sample(self.replay, self.batch_size)
        S = torch.tensor(np.array([e[0] for e in exp_replay]), dtype=torch.float)
        A = torch.tensor([e[1] for e in exp_replay], dtype=torch.int64)
        R = torch.tensor([e[2] for e in exp_replay], dtype=torch.float)
        S_ = torch.tensor(np.array([e[3] for e in exp_replay]), dtype=torch.float)
        D = torch.tensor([e[4] for e in exp_replay], dtype=torch.float)
        return S, A, R, S_, D

    def fill_memory(self):
        s = self.env.reset()
        for _ in range(self.replay_size):
            a = np.random.randint(4)
            s_, r, done, _ = self.env.step(a)
            self.replay.append([s,a,r,s_,done])
            s = s_
            if done:
                s = self.env.reset()

    def train(self, episodes, verbose=False):
        self.fill_memory()
        i = 0
        ep_rewards = np.zeros(episodes)
        for ep in tqdm(range(episodes)):
            ep_reward = 0
            done = False
            s = self.env.reset()
            ep_len = 0
            while not done and ep_len<1_000:

                ep_len += 1
                i = (i+1)%self.target_update

                # Pick action epsilon-greedy
                if np.random.random() < self.epsilon:
                    a = np.random.randint(4)
                else:
                    with torch.no_grad():
                        x = self.train_net(torch.tensor(s, dtype=torch.float))
                    a = x.numpy().argmax()

                # Remember outcome
                s_, r, done, _ = self.env.step(a)
                ep_reward += r
                self.replay.append([s,a,r,s_,done])
                s = s_

                # Sample batch from replay memory
                if len(self.replay) < self.batch_size:
                    continue
                S, A, R, S_, D = self.sample_from_exp_replay()
                Q = self.train_net(S).gather(1,A[:, None])
                with torch.no_grad():
                    Q_ = self.target_net(S_)
                y = R + (self.gamma * (Q_.max(1).values) * (1-D))

                # Perform gradient descent
                loss = self.loss(y[:,None], Q)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i==0:
                    # Update target model
                    self.target_net.load_state_dict(self.train_net.state_dict())

            ep_rewards[ep] = ep_reward
            if verbose and not ep%100:
                print('episode', ep)
                print('reward', total_r, '   epsilon', self.epsilon)
                print(self.train_net(torch.tensor(np.zeros(8), dtype=torch.float)))
            # Apply epsilon decay after every episode
            self.epsilon = max(self.epsilon*self.beta, self.e_min)
        self.env.close()
        return ep_rewards

    def test(self, trials, verbose=False, return_mean=True):
        scores = np.zeros(trials)
        ep_lens = np.zeros(trials)
        for i in tqdm(range(trials)):
            score = 0
            s = self.env.reset()
            done = False
            ep_len = 0
            while not done:
                ep_len += 1
                with torch.no_grad():
                    x = self.target_net(torch.tensor(s, dtype=torch.float))
                a = x.numpy().argmax()
                s_, r, done, _ = self.env.step(a)
                s = s_
                score += r
            scores[i] = score
            ep_lens[i] = ep_len
        if verbose:
            print(f"Average score from {trials} trials = {score.mean()}")
        self.env.close()
        if return_mean:
            return scores.mean(), ep_lens.mean()
        return scores, ep_len
