
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

class RRMC():
    def __init__(self):
        self.panda = rb.models.DH.Panda()

    def fkine(self):
        # Tip pose in world coordinate frame
        wTe = SE3(env._task.robot.arm.get_tip().get_position())*SE3.RPY(env._task.robot.arm.get_tip().get_orientation(), order='zyx')
        return wTe
    
    def target_pose(self):
        # Target pose in world coordinate frame
        wTt = SE3(env._task.target.get_position())*SE3.RPY(env._task.target.get_orientation(), order='zyx')
        print(env._task.target.get_orientation())
        return wTt
    
    def p_servo(self, gain=1):
        
        wTe = self.fkine()
        #print("wTe: ", wTe.t)
        wTt = self.target_pose()
        #print("wTt: ", wTt.t)
    
        # Pose difference
        eTt = wTe.inv() * wTt
        # Translational velocity error
        ev = eTt.t
        # Angular velocity error
        ew = eTt.rpy() * np.pi/180
        #ew = np.zeros(3)
        # Form error vector
        e = np.r_[ev, ew]
        #print("e: ", e)
        v = gain * e
        #print("v: ", v)
        return v
    
    def compute_action(self, gain=0.3):
        
        try:
            v = self.p_servo(gain)
            #v[3:] *= 10
            q = env._task.robot.arm.get_joint_positions()
            #print(q)
            #print(np.round(self.panda.jacobe(q), 2))
            action = np.linalg.pinv(self.panda.jacobe(q)) @ v
            #print("action: ", action)

        except np.linalg.LinAlgError:
            action = np.zeros(env_task.action_size)
            print('Fail')
        return action
        
        
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
        batch = random.sample(experience_dataset, 32)
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
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.wrist_camera.rgb = True
obs_config.joint_positions  = True

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env_task = Environment(action_mode, '', obs_config, False)
env_task.launch()

env = env_task.get_task(ReachTarget)
agent = Agent()

env._task.robot.arm.set_joint_forces([1000,1000,1000,1000,1000,1000,1000])


obs = env.reset()
control_prior = RRMC()
criterion = nn.MSELoss()
optimizer = optim.SGD(agent.pi.parameters(), lr=0.01, momentum=0.9)
total_steps = 10000000
episode_length = 150
experience_dataset = []

descriptions, state = env.reset()
obs = state.wrist_rgb


control_prior2 = joint_velocity_controller(env._task.robot.arm)
control_prior2.set_target(env._task.target)


for i in range(total_steps):
    if i%episode_length == 0 and i > 100:
        descriptions, state = env.reset()
        control_prior2.set_target(env._task.target)
        #agent.train()
    #action = agent.get_action(obs)
    #action = np.append(control_prior.compute_action(0.1), 0.0)
    action = np.append(control_prior2.compute_action(gain=0.8), 1.0)
    #action = np.zeros(8)
    next_state, reward, done = env.step(action)

    nobs = state.wrist_rgb
    #experience_dataset.append([control_prior.act(obs), obs])
    experience_dataset.append([action, obs])
    obs = nobs
    
print('Done')
env.shutdown()
    