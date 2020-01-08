import gym
import pybulletgym
import pybulletgym.envs
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import time

env = gym.make("FrozenLake-v0")
env.reset()
env.render()

"""
Action space:
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
"""
state = env.reset();


##TestPolicy
def TestPolicy (environment, policy, trials = 100):
    """
    Evaluate the policy 
    :param environment: the environment
    :param policy: the input policy
    :param trials: the no of trials
    """   
    success = 0;
    rewards = 0;
    for i in range(trials):
        terminated = False;
        state = environment.reset();
        #print("state", i, state)
        while not terminated:
            action = policy[state];
            next_state, reward, terminated, info = environment.step(action);
            rewards = rewards + reward;
            state = next_state;
            if (terminated and reward == 1):
                success = success + 1;
    av_reward =  rewards/trials;
    success = success/trials;
    return (av_reward,success)   

#taking a random deterministic policy
policy_rand = np.random.randint(4, size=env.nS)

#assigning the policy given in the question
policy_ques = np.zeros(env.nS)
states = np.arange(0,16)
policy_ques[states] = np.mod(states+1, 4)
print (policy_ques)

output = TestPolicy (env, policy_ques)
print(output)

#learn model
def LearnModel (environment, samples = 100000):
    """
    Learn the transition probablities (P(s'|a,s)) and the R(s',a, s)
    :param environment: environment
    :param currentState: state for which the table to be determined
    :param samples: no of samples
    """  
    init_state = environment.reset()
    dict_action_nextState = {}
    dict_reward = {}
    num_visited = {}
    for states in range(16):
        dict_action_nextState[states] = np.zeros(shape = (env.nA, env.nS))
        dict_reward[states] = np.zeros(shape = (env.nA, env.nS))
        #num_visited[states] = np.zeros(shape = (env.nA, env.nS))
    
    for iters in range(samples):    
        #for i in range(0,4):
        action = np.random.randint(4)
        next_state, reward, terminated, info = environment.step(action);
        dict_action_nextState[init_state][action][next_state] += 1
        dict_reward[init_state][action][next_state] += reward
        init_state = next_state
        #num_visited[init_state][action][:] += 1
        if (terminated):
            #take just one more action so that the initial state is the terminated state
            act = np.random.randint(4)
            next_state, reward, terminated, info = environment.step(act);
            dict_action_nextState[init_state][act][next_state] += 1
            init_state = environment.reset();
            
            

    
    for states in range(16):

        for row in range (dict_action_nextState[states].shape[0]):
            dict_action_nextState[states][row] = dict_action_nextState[states][row]/np.sum(dict_action_nextState[states][row])
            dict_reward[states] = dict_reward[states]/np.sum(dict_action_nextState[states][row])
    
    return dict_action_nextState, dict_reward


dict_action_nextState, dict_reward = LearnModel (env,20000)
print("========================ACTION_NEXTSTATE==============================")
print(dict_action_nextState)
print("========================REWARD========================================")
print(dict_reward)

def Policy_Evaluation(environment, policy, discount_factor = 1, theta = 1e-9, max_iterations = 1e9):
    V = np.zeros(environment.nS)
    evaluate_iter = 0
    for i in range(int(max_iterations)):
        delta = 0
        evaluate_iter += 1
        for state in range(environment.nS):
            v = 0
            for next_state in range(environment.nS):
                v += dict_action_nextState[state][int(policy[state])][next_state]*(dict_reward[state][int(policy[state])][next_state] + discount_factor * V[next_state])
            
            #calculate the delta change of value function
            delta = max(delta, np.abs(V[state] -  v))
            #update the value function
            V[state] = v
            
        # Terminate if value change is insignificant
        if delta < theta:
            #print(f'Policy evaluated in {evaluate_iter} iterations.')
            return V
    
    print(delta)
    return V

def Lookahead(environment, state, V, discount_factor):
    action_values = np.zeros(environment.nA)
    for action in range(environment.nA):
        for next_state in range(environment.nS):
            action_values[action] +=  dict_action_nextState[state][action][next_state] * (dict_reward[state][action][next_state] + discount_factor * V[next_state])
    return action_values

def policy_iteration(environment, policy, discount_factor=1.0, max_iterations=50):
    iters = []
    evaluation = []
    evaluate_iter = 0
    flag = 0
    for i in range(max_iterations):
        #print(f'Policy in {i} iter. is {policy}')
        stable_policy = True
        evaluate_iter+= 1
        V = Policy_Evaluation(environment, policy, discount_factor = discount_factor)
        #Go through each state and try to improve the action taken
        for state in range(environment.nS):
            curr_action = policy[state]
            #now evaluate every other action
            action_values = Lookahead(environment, state, V, discount_factor);
            # a better action
            best_action = np.argmax(action_values)
            #greedy update
            policy[state] = best_action
            if(best_action != curr_action):
                stable_policy = False #making the stable false if any action changes for the state
        eval = TestPolicy(environment, policy, trials = 100)
                
        iters.append(i)
        evaluation.append(eval[1])
        if (stable_policy and flag == 0) :
            print(f'Policy converged in  {evaluate_iter} iterations.')
            flag = 1
            #plt.plot(iters,evaluation, color='b');
            #plt.show()
            #return policy

    plt.plot(iters,evaluation, color='b');
    plt.xlabel("iterations")
    plt.ylabel("test_policy")
    plt.show() 
    return policy

policy_ques = np.zeros(env.nS)
states = np.arange(0,16)
policy_ques[states] = np.mod(states+1, 4)
Policy = policy_iteration(env, policy_ques)
print(Policy)

#Start with random policies
policy_rand1 = np.random.randint(4, size=env.nS)
Policy = policy_iteration(env, policy_rand1)
print(Policy)

policy_rand2 = np.random.randint(4, size=env.nS)
Policy = policy_iteration(env, policy_rand2)
print(Policy)

def Value_iteration(environment, discount_factor = 1.0, theta = 1e-9, max_iterations=50):
    V = np.zeros(environment.nS)
    policy = np.zeros(environment.nS)
    evaluate_iter = 0
    iters = []
    evaluation = []
    for i in range(max_iterations):
        evaluate_iter+= 1
        delta = 0
        for state in range(environment.nS):
            action_value = Lookahead(environment, state, V, discount_factor)
            best_action_value = np.max(action_value)
            best_action = np.argmax(action_value)
            delta = max(delta, np.abs(V[state] - best_action_value))
            V[state] = best_action_value
            policy[state] = best_action
        eval = TestPolicy(environment, policy, trials = 100);
        iters.append(i)
        evaluation.append(eval[1])

        if(delta < theta):
            print(f'Value converged in  {evaluate_iter} iterations.')
            plt.plot(iters,evaluation, color='b');
            plt.show()
            return policy
    
    plt.plot(iters,evaluation, color='b');
    plt.xlabel("iterations")
    plt.ylabel("test_policy")
    plt.show()
    return policy

policy = Value_iteration(env)
print(policy)

def choose_action(state, epsilon, Q):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def Q_learning(environment, gamma = 0.99, alpha = 0.05, total_episodes = 5000, max_steps = 50):
    Q = np.zeros((environment.nS, environment.nA))
    policy = np.zeros(environment.nS)
    episodes = []
    evaluations = []
    for episode in range(total_episodes):
        #print(episode)
        state = environment.reset()
        t = 0
    
        while t < max_steps:
        
            action = choose_action(state, 1 - episode/5000, Q)  
            next_state, reward, done, info = environment.step(action)  
            predict = Q[state,action]
            target = reward + gamma * np.max(Q[next_state,:])
            Q[state,action] = Q[state, action] + alpha * (target - predict)
            state = next_state
        
            #start with a new episode
            if done:
                #determine the policy
                policy = np.argmax(Q, axis = 1)
                evaluation = TestPolicy(environment, policy, trials = 100);
  
                break

            
            t+= 1
            
        policy = np.argmax(Q, axis = 1)
        if(episode%100 == 0):
            evaluation = TestPolicy(environment, policy, trials = 100);
            episodes.append(episode)
            evaluations.append(evaluation[1])
    

    plt.plot(episodes,evaluations, color='b');
    title = "model: alpha ="+ str(alpha) + "gamma ="+ str(gamma);
    
    plt.title(title)
    plt.xlabel("episodes")
    plt.ylabel("test_policy")
    plt.show()
    return Q

gamma_list = [0.90, 0.95, 0.99]
alpha_list = [0.05, 0.1, 0.25, 0.5]

for i in range(0,len(alpha_list)):
    Qans = Q_learning(env, gamma = 0.99, alpha = alpha_list[i])
    policy = np.argmax(Qans, axis = 1 )
    print(f'The policy for alpha value {alpha_list[i]} and gamma value 0.99 is {policy}.')
    

for i in range(0,len(gamma_list)):
    Qans = Q_learning(env, gamma = gamma_list[i], alpha = 0.05)
    policy = np.argmax(Qans, axis = 1)
    print(f'The policy for alpha value 0.05 and gamma value {gamma_list[i]} is {policy}.')

def Q_learning_opt(environment, gamma = 0.99, alpha = 0.05, explore = 1, total_episodes = 5000, max_steps = 50):
    Q = np.zeros((environment.nS, environment.nA))
    policy = np.zeros(environment.nS)
    episodes = []
    evaluations = []
    for episode in range(total_episodes):
        #print(episode)
        state = environment.reset()
        t = 0
    
        while t < max_steps:
            action = choose_action(state, explore, Q)  
            next_state, reward, done, info = environment.step(action)  
            predict = Q[state,action]
            target = reward + gamma * np.max(Q[next_state,:])
            Q[state,action] = Q[state, action] + alpha * (target - predict)
            #print(Q)
            state = next_state
        
            #start with a new episode
            if done:
                #determine the policy
                policy = np.argmax(Q, axis = 1)
                evaluation = TestPolicy(environment, policy, trials = 100);
                break

            #time.sleep(0.1)
            t+= 1
            #print(f'Value t is   {t}')
        policy = np.argmax(Q, axis = 1)
        if(episode%100 == 0):
            evaluation = TestPolicy(environment, policy, trials = 100);
            episodes.append(episode)
            evaluations.append(evaluation[1])
    
    plt.plot(episodes,evaluations, color='b');
    title = "model: alpha ="+ str(alpha) + "gamma ="+ str(gamma);

    plt.title(title)
    plt.xlabel("episodes")
    plt.ylabel("test_policy")
    plt.show()
    return Q

Qans = Q_learning_opt(env, gamma = 0.99, alpha = 0.05)
policy = np.argmax(Qans, axis = 1)
print(f'The policy for alpha value 0.05 and gamma value 0.99 is {policy}.')

Qans = Q_learning_opt(env, gamma = 0.99, alpha = 0.05, explore = 0.9)
policy = np.argmax(Qans, axis = 1)
print(f'The policy for alpha value 0.05 and gamma value 0.99 is {policy}.')

Qans = Q_learning_opt(env, gamma = 0.99, alpha = 0.05, explore = 0.5)
policy = np.argmax(Qans, axis = 1)
print(f'The policy for alpha value 0.05 and gamma value 0.99 is {policy}.')
