
# DAgger Implementation for RLBench
# Paper: https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf
# Author: Krishan Rana

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PushButtons
import pdb
import numpy as np
from spatialmath import SE3, SO3
import pdb
import roboticstoolbox as rb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from velocity_controller import joint_velocity_controller
import wandb
import math


class Model(nn.Module):
    
    def __init__(self, act_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(2704, 520)
        self.fc2 = nn.Linear(520,108)
        self.fc3 = nn.Linear(108,54)
        self.fc4 = nn.Linear(54, act_dim)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x
    
class Agent:
    
    def __init__(self):
        self.pi = Model(env_task.action_size).to(device)
        
    def train(self):
        print("Training...")
        batch = random.sample(experience_dataset, batch_size)
        batch_obs = torch.as_tensor([demo[1] for demo in batch], dtype=torch.float32).to(device).permute(0,3,1,2)
        predicted_actions = self.pi(batch_obs)
        ground_truth_actions = torch.as_tensor([demo[0] for demo in batch], dtype=torch.float32).to(device).detach()
        loss = criterion(predicted_actions, ground_truth_actions)
        loss.backward()
        optimizer.step()
        
    def get_action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            obs = obs.permute(0,3,1,2)
            act = self.pi(obs).cpu().numpy()[0]
            return act

    def evaluate(self):
        success = 0
        for i in range(5):
            descriptions, state = env.reset()
            obs = state.wrist_rgb
            done = False
            for j in range(150):
                a = agent.get_action(obs)
                next_state, reward, done = env.step(a)
                if done:
                    success += 1
                    break
                obs = next_state.wrist_rgb
        wandb.log({"success_rate":(success/5)})
        return

def decay(k, x0, x):
    b = 1 / (1 + math.exp(k*(x - x0))) 
    return b

wandb.init(project="DAgger")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.wrist_camera.rgb = True
obs_config.joint_positions  = True
obs_config.image_size = (64, 64)

# SETUP
# Environment
action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env_task = Environment(action_mode, '', obs_config, False)
env_task.launch()
env = env_task.get_task(ReachTarget)
descriptions, state = env.reset()
obs = state.wrist_rgb
done = False
episode_num = 0
#env._task.robot.arm.set_joint_forces([1000,1000,1000,1000,1000,1000,1000])

# Agent
agent = Agent()
criterion = nn.MSELoss()
optimizer = optim.SGD(agent.pi.parameters(), lr=0.01, momentum=0.9)
total_steps = int(10e6)
episode_length = 150
batch_size = 32
evaluate_after = 10
experience_dataset = []

# Control Prior
control_prior = joint_velocity_controller(env._task.robot.arm)
control_prior.set_target(env._task.target)

# RUN
for i in range(total_steps):

    if done and i > 100: #i%episode_length == 0 and i > 100:
        agent.train()
        episode_num += 1
        if episode_num%evaluate_after == 0:
            print("Evaluating...")
            agent.evaluate()
        descriptions, state = env.reset()
        obs = state.wrist_rgb
        control_prior.set_target(env._task.target) 

    cq = env._task.robot.arm.get_joint_positions()
    tq = control_prior.target_q

    # Compute action using combinatorial approach shown in paper
    beta = decay(0.00004, 200000, i)
    expert_action = control_prior.compute_action(gain=0.8)
    policy_action = agent.get_action(obs)
    action = (beta * expert_action) + (1-beta)*policy_action

    next_state, reward, done = env.step(action)
    nobs = next_state.wrist_rgb
    experience_dataset.append([control_prior.recompute_action(cq, tq, gain=0.8), obs])
    obs = nobs

    wandb.log({"beta":beta})
    
print('Done')
env.shutdown()
    