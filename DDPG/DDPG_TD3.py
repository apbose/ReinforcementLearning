import gym
import pybulletgym
import pybulletgym.envs
import numpy as np
import math
import matplotlib.pyplot as plt
import queue
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init = False)
env.reset()

class Actor(nn.Module) :
    def __init__(self, state_dim, action_dim, hidden_size_one, hidden_size_two):
        
        super(Actor, self).__init__()
        self.input_size = state_dim;
        self.hidden_size_one = hidden_size_one;
        self.hidden_size_two = hidden_size_two;
        self.output_size = action_dim
        
        self.l1 = nn.Linear(self.input_size, self.hidden_size_one, bias = False)
        self.l2 = nn.Linear(self.hidden_size_one, self.hidden_size_two, bias = False)
        self.l3 = nn.Linear(self.hidden_size_two, self.output_size, bias = False)
        
        self.model = torch.nn.Sequential(
                                    self.l1,
                                    nn.ReLU(),
                                    #nn.Tanh(),
                                    self.l2,
                                    nn.ReLU(),
                                    self.l3,
                                    nn.Tanh()
                                    )
        self.model.apply(self.weights_init_uniform)

        
    # takes in a module and applies the specified weight initialization
    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        # apply a uniform distribution to the weights and a bias=0
        if classname.find('Linear') != -1:
            m.weight.data.uniform_(-0.003, 0.003)
            #m.bias.data.fill_(0)
    
    def forward (self, state):

        
        
        return self.model(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size_one, hidden_size_two):
        
        super(Critic, self).__init__()
        self.input_size = (state_dim + action_dim);
        self.hidden_size_one = hidden_size_one;
        self.hidden_size_two = hidden_size_two;
        self.output_size = 1
        
        self.l1 = nn.Linear(self.input_size, self.hidden_size_one, bias = False)
        self.l2 = nn.Linear(self.hidden_size_one, self.hidden_size_two, bias = False)
        self.l3 = nn.Linear(self.hidden_size_two, self.output_size, bias = False)
        self.model_one = torch.nn.Sequential(
                                    self.l1,
                                    nn.ReLU(),
                                    #nn.Tanh(),
                                    self.l2,
                                    nn.ReLU(),
                                    self.l3,
                                    nn.Tanh()
                                    )
        self.model_one.apply(self.weights_init_uniform)
        
        self.l4 = nn.Linear(self.input_size, self.hidden_size_one, bias = False)
        self.l5 = nn.Linear(self.hidden_size_one, self.hidden_size_two, bias = False)
        self.l6 = nn.Linear(self.hidden_size_two, self.output_size, bias = False)
        self.model_two = torch.nn.Sequential(
                                    self.l4,
                                    nn.ReLU(),
                                    #nn.Tanh(),
                                    self.l5,
                                    nn.ReLU(),
                                    self.l6,
                                    nn.Tanh()
                                    )
        self.model_two.apply(self.weights_init_uniform)
    
    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        # apply a uniform distribution to the weights and a bias=0
        if classname.find('Linear') != -1:
            m.weight.data.uniform_(-0.0003, 0.0003)
            #m.bias.data.fill_(0)
            
    def Q1(self, state, action):
        stateAction = torch.cat([state, action], 1)
        return self.model_one(stateAction)
    
    def forward (self, state, action):
        
        stateAction = torch.cat([state, action], 1)
        return self.model_one(stateAction), self.model_two(stateAction)


class replayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size;
        self.buffer = deque(maxlen = buffer_size)
        
    def push (self, state, action, next_state, reward, done):
        samples = (state, action, next_state, reward, done)
        self.buffer.append(samples)
    
    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        done_batch = []
        
        batch_data = random.sample(self.buffer, batch_size)
        
        for samples in batch_data:
            state, action, next_state, reward, done = samples
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        return (state_batch, action_batch, next_state_batch, reward_batch, done_batch)
    
    def __len__(self):
        return len(self.buffer)

class DDPG():
    def __init__(self, 
                 env, 
                 action_dim, 
                 state_dim, 
                 actor, 
                 critic, 
                 actor_target, 
                 critic_target, 
                 noise = 1,
                 d_param = 0.001,
                 critic_lr = 0.0003, 
                 actor_lr = 0.0003, 
                 gamma = 0.99, batch_size = 100, buffer_size = 10000,pol_freq = 2):
        
        """        
        param: env: An gym environment        
        param: action_dim: Size of action space        
        param: state_dim: Size of state space
        param: actor: actor model
        param: critic: critic model
        param: critic_lr: Learning rate of the critic        
        param: actor_lr: Learning rate of the actor        
        param: gamma: The discount factor        
        param: batch_size: The batch size for training        
        """
        

        self.env = env
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.d = d_param
        self.noise = noise
        self.pol_freq = pol_freq
        
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr= self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        
        self.iterations = []
        self.return_reward = []
        
        self.replay_buffer = replayBuffer(buffer_size)
        self.loss = nn.MSELoss()
    
    def updateQpolicy(self, batch_size, iteration):
        states, actions, state_next,rewards, _ = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).reshape([batch_size,1])
        #rewards = torch.FloatTensor(rewards)
        state_next = torch.FloatTensor(state_next)
        #print("states size", states.size())
        Q_presone, Q_prestwo = self.critic.forward(states, actions)
        #Q_pres = critic.forward(states, actions)
        action_next = actor_target.forward(state_next) 
        Q_nextone, Q_nexttwo = self.critic_target.forward(state_next, action_next.detach())#while doing loss.backward we dont want target_policy parameters to be updated
        #print("rewards", rewards.size())#100
        #print("q_next",Q_next.size())#100*1
        Q_nextchosen = torch.min(Q_nextone, Q_nexttwo)
        Q_nexttarget = rewards + Q_nextchosen * self.gamma
        #wrt Q parameter maps s and actions to theQ value
        #print("q_nexttaget",Q_nexttarget.size())#100*100
        #print("q_pres",Q_pres.size())#100*1
        criticLoss = self.loss(Q_nexttarget, Q_presone) + self.loss(Q_nexttarget, Q_prestwo)
        
        
        
        #update the Q paramters which maps states to actions to the Q value
        self.critic_optimizer.zero_grad();
        criticLoss.backward();
        self.critic_optimizer.step();
        
        if(iteration % self.pol_freq == 0):
            #wrt policy parameter, maps states to actions
            actorLoss = -1 * self.critic.Q1(states, actor.forward(states)).mean()
        
            #update thw policy parameters which updates the states to actions
            self.actor_optimizer.zero_grad();
            actorLoss.backward();
            self.actor_optimizer.step();
        
            #update the target network weights with the original network weights
            for tar_param, src_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                tar_param.data.copy_(self.d * src_param.data + (1.0 - self.d) * tar_param.data)
    
        for tar_param, src_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            tar_param.data.copy_(self.d * src_param.data + (1.0 - self.d) * tar_param.data)
    
    def selectAction(self, state):
        #state = torch.FloatTensor(state)
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0]
        return action
    
    
    def train(self, epochs):
        total_reward = 0
        for iterationNo in range(epochs):
            state = env.reset()
            batch_reward = 0
            
            steps = 0
            '''
            while(steps < self.batch_size):
                steps += 1
                action = self.selectAction(state)
                if(self.noise):
                    #mean = torch.zeros(2);
                    #variance = torch.diag([0.1, 0.1])
                    #c = MultivariateNormal(mean, variance)
                    #noise = c.sample()
                    noise = np.random.normal(0, 0.1) 
                    action[0]+= noise
                    action[1]+= noise
                new_state, reward, done, _ = env.step(action)
                
                batch_reward += reward
                total_reward += reward
                self.replay_buffer.push(state, action, new_state, reward, done)
                state = new_state

                
                
                if(done == True):
                    break;
            '''
            #fill up the buffer
            while(len(self.replay_buffer)< self.batch_size):
                action = self.selectAction(state)
                if(self.noise):
                    #mean = torch.zeros(2);
                    #variance = torch.diag([0.1, 0.1])
                    #c = MultivariateNormal(mean, variance)
                    #noise = c.sample()
                    noise = np.random.normal(0, 0.1) 
                    action[0]+= noise
                    action[1]+= noise
                new_state, reward, done, _ = env.step(action)
                if(done == True):
                    state= env.reset()
                
                batch_reward += reward
                total_reward += reward
                self.replay_buffer.push(state, action, new_state, reward, done)
                state = new_state

            #if(len(self.replay_buffer) >= self.batch_size):
            #if(iterationNo%self.batch_size == 0 and len(self.replay_buffer)>= self.batch_size):
            action = self.selectAction(state)
            new_state, reward, done, _ = env.step(action)
            if(done == True):
                state = env.reset()
            batch_reward += reward
            total_reward += reward
            self.replay_buffer.push(state, action, new_state, reward, done)
            state = new_state
            
            self.updateQpolicy(self.batch_size, iterationNo)
            if((iterationNo % 1000 == 0 and iterationNo!=0) or iterationNo == 1):
                self.iterations.append(iterationNo)
                self.return_reward.append(total_reward/iterationNo)
                print("iteration No is", iterationNo, "reward is", total_reward/iterationNo)
                #self.return_history.append(batch_reward)
            
            if(iterationNo%2000 == 0 and iterationNo!= 0):
                fileName = "model_td3"+str(iterationNo)
                torch.save(self.actor.state_dict(), fileName)

num_states = 8
num_actions = 2

actor = Actor(num_states, num_actions, 400, 300)
actor_target = Actor(num_states, num_actions, 400, 300)

critic = Critic(num_states, num_actions, 400, 300)
critic_target = Critic(num_states, num_actions, 400, 300)

for tar_param, src_param in zip(actor_target.parameters(), actor.parameters()):
    tar_param.data.copy_(src_param.data)
    
for tar_param, src_param in zip(critic_target.parameters(), critic.parameters()):
    tar_param.data.copy_(src_param.data)



#ddpgLinkArm = DDPG(env, num_actions, num_states, actor, critic, actor_target, critic_target, noise )
ddpgLinkArm = DDPG(env, num_actions, num_states, actor, critic, actor_target, critic_target)
ddpgLinkArm.train(50000)

del ddpgLinkArm.iterations[0]
del ddpgLinkArm.return_reward[0]
plt.plot(ddpgLinkArm.iterations,ddpgLinkArm.return_reward, color='b');
plt.xlabel("iterations")
plt.ylabel("return_history")
plt.show()
