import torch
import gym
import numpy as np 
import random
import torch.nn as nn
import copy
import collections
from tensorboardX import SummaryWriter

from torch.autograd import Variable




class Q_network(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Q_network, self).__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 50), nn.ReLU(), 
                                 nn.Linear(50, act_dim))
        
        self.net[0].weight.data.normal_(0, 0.1)
        self.net[2].weight.data.normal_(0, 0.1)

    def forward(self, obs):
        return self.net(obs)
        

class DQN():
    def __init__(self, env):
        self.env = env
        self.act_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self.buffer_size = 2000
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.batch = 32
        self.lr = 0.01
        self.seed = 0
        self.episodes = 1000000
        self.gamma = 0.95
        self.target_update_freq = 1000
        self.learn_step_counter = 0
        self.timeout = 500

        
        self.Q = Q_network(self.obs_dim, self.act_dim)
        self.Q_target =  Q_network(self.obs_dim, self.act_dim)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.replay_buffer = collections.deque(maxlen=self.buffer_size)
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.writer = SummaryWriter()

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        env.seed(self.seed)


    def train(self):
        # target parameter update
        if self.learn_step_counter % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        batch = random.sample(self.replay_buffer, self.batch)

        b_o   = np.array([b[0] for b in batch]).astype(np.float32)
        b_a   = np.array([b[1] for b in batch]).astype(np.float32)
        b_r   = np.array([b[2] for b in batch]).astype(np.float32)
        b_no  = np.array([b[3] for b in batch]).astype(np.float32)
        b_d   = np.array([b[4] for b in batch]).astype(np.float32)


        # to torch
        b_s  = torch.FloatTensor(torch.from_numpy( b_o))
        b_a  = torch.LongTensor(torch.from_numpy( b_a.astype(int)).view(-1,1))
        b_r  = torch.FloatTensor(torch.from_numpy( b_r).view(-1,1))
        b_s_ = torch.FloatTensor(torch.from_numpy(b_no))
        b_d = torch.FloatTensor(torch.from_numpy( b_d).view(-1,1))


        # q_eval w.r.t the action in experience
        # Gathers all the q values pertaining to the action taken
        q_eval = self.Q(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.Q_target(b_s_)     # detach from graph, don't backpropagate

        est = self.Q_target(b_s_).gather(1,torch.argmax(self.Q(b_s_), 1).view(self.batch,1))

        q_target = b_r + self.gamma * est*(1-b_d)   # shape (batch, 1)
        loss = torch.mean((q_eval - q_target.detach())**2)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()


    def learn(self):

        total_steps = 0

        for ep in range(self.episodes):

            obs = self.env.reset()
            done = False
            steps = 0

            for t in range(self.timeout):
                if np.random.random() < self.epsilon:
                    act = self.env.action_space.sample() 
                else:
                    with torch.no_grad(): 
                        act = np.argmax(self.Q(torch.FloatTensor(obs)).data.numpy())
                
               # act = self.choose_action(obs)

                nobs, rew, done, _ = self.env.step(act)
                
                if done and t < self.timeout-1: rew = -1
                else: rew = 0
                # Append to replay buffer
                self.replay_buffer.append((obs, act, rew, nobs, done))
                
                obs = nobs

                # Train the agent
                if len(self.replay_buffer) > self.batch:
                    loss = self.train()
                    self.writer.add_scalar('dqn/loss', loss, total_steps)

                steps += 1
                total_steps += 1
                self.env.render()

                if done: break
                
            self.writer.add_scalar('dqn/len', steps, ep)
            self.epsilon *= self.epsilon_decay


    def evaluate(self):
        pass

    
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    agent = DQN(env)
    agent.learn()
