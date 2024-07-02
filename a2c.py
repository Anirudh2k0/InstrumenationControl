import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from riseEnv import RiseEnv
from networks import Actor,Critic

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

# helper function to convert numpy arrays to tensors
def t(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    
    return torch.from_numpy(x).float()

def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1-done)
        discounted.append(ret)
    
    return discounted[::-1]

def process_memory(memory, gamma=0.99, discount_rewards=True):
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []

    for action, reward, state, next_state, done in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        dones.append(done)
    
    if discount_rewards:
        if False and (dones[-1] == 0):
            rewards = discounted_rewards(rewards + [last_value], dones + [0], gamma)[:-1]
        else:
            rewards = discounted_rewards(rewards, dones, gamma)

    actions = t(actions).view(-1, 1)
    states = t(states)
    next_states = t(next_states)
    rewards = t(rewards).view(-1, 1)
    dones = t(dones).view(-1, 1)
    return actions, rewards, states, next_states, dones

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

class A2CLearner():
    def __init__(self, actor, critic, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    
    def learn(self, memory, steps, discount_rewards=True):
        actions, rewards, states, next_states, dones = process_memory(memory, self.gamma, discount_rewards)

        if discount_rewards:
            td_target = rewards
        else:
            print(f'next_states: {next_states}')
            td_target = rewards + self.gamma*critic(next_states)*(1-dones)
        value = critic(states)
        advantage = td_target - value

        # actor
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()
        
        actor_loss = (-logs_probs*advantage.detach()).mean() - entropy*self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()
        
        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        writer.add_histogram("gradients/actor",
                             torch.cat([p.grad.view(-1) for p in self.actor.parameters()]), global_step=steps)
        writer.add_histogram("parameters/actor",
                             torch.cat([p.data.view(-1) for p in self.actor.parameters()]), global_step=steps)
        self.actor_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        writer.add_histogram("gradients/critic",
                             torch.cat([p.grad.view(-1) for p in self.critic.parameters()]), global_step=steps)
        writer.add_histogram("parameters/critic",
                             torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=steps)
        self.critic_optim.step()
        
        # reports
        writer.add_scalar("losses/log_probs", -logs_probs.mean(), global_step=steps)
        writer.add_scalar("losses/entropy", entropy, global_step=steps) 
        writer.add_scalar("losses/entropy_beta", self.entropy_beta, global_step=steps) 
        writer.add_scalar("losses/actor", actor_loss, global_step=steps)
        writer.add_scalar("losses/advantage", advantage.mean(), global_step=steps)
        writer.add_scalar("losses/critic", critic_loss, global_step=steps)

class Runner():
    def __init__(self, env):
        self.env = env
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
    
    def reset(self):
        self.episode_reward = 0
        self.done = False
        self.state = self.env.reset()
    
    def run(self, max_steps, memory=None):
        if not memory: memory = []
        
        for i in range(max_steps):
            if self.done: self.reset()
           
            dists = actor(t([self.state]))
            
            actions = dists.sample().detach().data.numpy()
            actions_clipped = np.clip(actions, 550,850)

            next_state, reward, self.done, info = self.env.step(actions_clipped)
            memory.append((actions, reward, self.state, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward
            
            if self.done:
                self.episode_rewards.append(self.episode_reward)
                if len(self.episode_rewards) % 10 == 0:
                    print("episode:", len(self.episode_rewards), ", episode reward:", self.episode_reward)
                writer.add_scalar("episode_reward", self.episode_reward, global_step=self.steps)
                    
        
        return memory


dfOrig= pd.read_csv('Data/Synthetic Data.csv')
df = dfOrig.copy()
curr_vals = [700,4.2,1,1,20]
req_gr = 6.0
dfGR = df[df['GrowthRate'].between(0.95*req_gr, 1.05*req_gr)]
dfTemp = dfGR[dfGR['Temperature']>=curr_vals[0]]
env = RiseEnv(dfTemp,req_gr,curr_vals)
writer = SummaryWriter("runs/mish_activation")

# config
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
actor = Actor(state_dim, n_actions, activation=Mish)
critic = Critic(state_dim, activation=Mish)

learner = A2CLearner(actor, critic)
runner = Runner(env)


# steps_on_memory = 16
steps_on_memory = 64
episodes = 500
episode_length = 200
total_steps = (episode_length*episodes)//steps_on_memory

for i in range(total_steps):
    memory = runner.run(steps_on_memory)
    learner.learn(memory, runner.steps, discount_rewards=False)