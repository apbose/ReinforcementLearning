import gym
import pybulletgym
import pybulletgym.envs
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

env = gym.make("CartPole-v1")
env.reset()

learning_rate = 0.01
gamma = 0.99

#4 states, 2 actions
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.state_num = num_states
        self.action_num = num_actions
        
        self.l1 = nn.Linear(self.state_num, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_num, bias=False)
        
        self.gamma = gamma
        
        # Episode policy and reward history 
        self.policy_episode = Variable(torch.Tensor()) 
        self.reward_episode = []   #a hash containing the reward of steps of a particular episode
        # Overall return history
        self.return_history = []
        self.return_reward = []
        self.return_history_stepzero = []
        self.episodesPeriter = []
        



    def forward(self, x): 
        model = torch.nn.Sequential(
                                    self.l1,
                                    nn.Dropout(p=0.6),
                                    #nn.ReLU(),
                                    nn.Tanh(),
                                    self.l2,
                                    nn.Softmax(dim=-1)
                                    )
        return model(x)


class CartPole ():
    

    
    def __init__(self, env, policy, part, totsteps, iterationsNo, learning_rate, gamma):
        self.env = env
        self.policy = policy
        self.part = part
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.iterationsNo = iterationsNo
        self.totsteps = totsteps
        
        self.iterations = []
        
        self.polHist_allepisodes = Variable(torch.Tensor())
        self.rewHist_allepisodes = Variable(torch.Tensor())
        self.rewHist_allepisodes_mod = Variable(torch.Tensor())
    
    def select_action(self, state):
        #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        state = torch.from_numpy(state).type(torch.FloatTensor)
        act_prob = self.policy(state)
        c = Categorical(act_prob)
        action = c.sample()

        if len(self.policy.policy_episode) > 0:
            self.policy.policy_episode = torch.cat([self.policy.policy_episode, c.log_prob(action).reshape(1)])
        else:
            self.policy.policy_episode = (c.log_prob(action).reshape(1))
        return action

    def rewardFunction(self, polHistory, rewHistory, part = 1):

        Rewards_tot = 0
        rewards = []
        for Reward in rewHistory[::-1]:
            Rewards_tot = Reward + self.policy.gamma * Rewards_tot
            rewards.insert(0, Rewards_tot)

        if(self.part == 1):
            #gainPerStep = rewards[0]
            gainPerStep = sum(rewHistory)
            gainZeroStep = rewards[0]
            reward = rewards[0]*torch.sum(polHistory)
            


        if(self.part == 2):
            gainPerStep = sum(rewHistory)
            gainZeroStep = rewards[0]
            rewards = torch.FloatTensor(rewards)
            reward = torch.sum(torch.mul(polHistory, rewards))
        
        if(self.part == 3):
            gainPerStep = sum(rewHistory)
            gainZeroStep = rewards[0]
            rewards = torch.FloatTensor(rewards)
            #rewards = (rewards - rewards.mean())/ (rewards.std())
            #reward = torch.sum(torch.mul(polHistory, rewards))
            
            #for i in range(0,len(polHistory)):
            #print(len(self.polHist_allepisodes))
            #print(len(self.rewHist_allepisodes))
            
            #if self.polHist_allepisodes.size(0) > 0:
            if (len(self.polHist_allepisodes) > 0):
                #print(self.polHist_allepisodes.shape)
                #print(type(polHistory))
                self.polHist_allepisodes = torch.cat([self.polHist_allepisodes, polHistory])
                self.rewHist_allepisodes = torch.cat([self.rewHist_allepisodes, rewards])
            else:
                self.polHist_allepisodes = polHistory
                self.rewHist_allepisodes = rewards
            #rewards = (rewards - rewards.mean())/ (rewards.std())
            #reward = torch.sum(torch.mul(polHistory, rewards))
            
            #self.rewHist_allepisodes = (self.rewHist_allepisodes - self.rewHist_allepisodes.mean())/ (self.rewHist_allepisodes.std())
            self.rewHist_allepisodes_mod = (self.rewHist_allepisodes - self.rewHist_allepisodes.mean())/ (self.rewHist_allepisodes.std())
            reward = torch.sum(torch.mul(self.polHist_allepisodes, self.rewHist_allepisodes_mod))
            #print(reward)
        
        
        return reward, gainPerStep, gainZeroStep
    
    
    def update_policy(self, reward, retTraj_tot, retTraj_tot_stepzero, episodes_iter):
        #print("in update")
        # Update network weights
        self.optimizer.zero_grad()
        #print(reward)
        reward.backward()
        self.optimizer.step()
        self.policy.return_history.append(retTraj_tot)
        #self.policy.return_reward.append(reward)
        self.policy.return_history_stepzero.append(retTraj_tot_stepzero)
        self.policy.episodesPeriter.append(episodes_iter)
    
    def reinforceAlgo(self):
    #running_reward = 10
    
        for iter in range(self.iterationsNo):
            self.polHist_allepisodes = Variable(torch.Tensor())
            self.rewHist_allepisodes = Variable(torch.Tensor())
            #print("iteration no",iter)
            steps = 0
            state = env.reset() # Reset environment, starting state recorded
            done = False
            episodes = 0;
            rewardFunc = Variable(torch.FloatTensor([0])) 
            rewardEpisode = Variable(torch.FloatTensor()) 
            retTraj_tot = 0
            retStepzero_tot = 0
            #optimizer.zero_grad()
            while(steps < self.totsteps):
                steps += 1;
                action = self.select_action(state)
                # Step through environment using chosen action
                state, reward, done, _ = env.step(action.item())
                #env.render()
            
            # Save reward
                self.policy.reward_episode.append(reward)
                if (done == True):
                    rewardEpisode, retTraj, retStepzero = self.rewardFunction(self.policy.policy_episode, self.policy.reward_episode, self.part)
                    if(self.part != 3):
                        rewardFunc += rewardEpisode
                    retTraj_tot += retTraj
                    retStepzero_tot += retStepzero
                    #reset the environment again
                    self.policy.policy_episode = Variable(torch.Tensor()) 
                    self.policy.reward_episode = []
                    state = env.reset()
                    done = False
                    episodes += 1
                
            if(self.part == 3):
                 rewardFunc += rewardEpisode
            if(episodes > 0):
                self.update_policy(-1 * rewardFunc/episodes, retTraj_tot/episodes,retStepzero_tot/episodes, episodes)
            else:
                self.update_policy(-1 * rewardFunc, retTraj_tot, retStepzero_tot, 0)
            self.iterations.append(iter)


policy = PolicyNetwork()
cartPole = CartPole(env= env, policy = policy, part = 1, totsteps = 500, iterationsNo = 200, learning_rate = 0.01, gamma = 0.99)
cartPole.reinforceAlgo()

plt.plot(cartPole.iterations,cartPole.policy.return_history_stepzero, color='b');
plt.xlabel("iterations")
plt.ylabel("avreturn_trajectory")
plt.show()


plt.plot(cartPole.iterations,cartPole.policy.episodesPeriter, color='b');
plt.xlabel("iterations")
plt.ylabel("episodes")
plt.show()


policy = PolicyNetwork()
cartPole = CartPole(env= env, policy = policy, part = 2, totsteps = 500, iterationsNo = 200, learning_rate = 0.01, gamma = 0.99)
cartPole.reinforceAlgo()



plt.plot(cartPole.iterations,cartPole.policy.return_history_stepzero, color='b');
plt.xlabel("iterations") 
plt.ylabel("avreturn_trajectory")
plt.show()


plt.plot(cartPole.iterations,cartPole.policy.episodesPeriter, color='b');
plt.xlabel("iterations")
plt.ylabel("episodes")
plt.show()

policy = PolicyNetwork()
cartPole = CartPole(env= env, policy = policy, part = 3, totsteps = 500, iterationsNo = 200, learning_rate = 0.01, gamma = 0.99)
cartPole.reinforceAlgo()


plt.plot(cartPole.iterations,cartPole.policy.return_history_stepzero, color='b');
plt.xlabel("iterations")
plt.ylabel("avreturn_trajectory")
plt.show()


plt.plot(cartPole.iterations,cartPole.policy.episodesPeriter, color='b');
plt.xlabel("iterations")
plt.ylabel("episodes")
plt.show()
   
