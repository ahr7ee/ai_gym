import gym
import numpy as np
env = gym.make('CartPole-v0')
best_reward=0
parameters = np.random.rand(4) * 2 - 1
for i_episode in range(300):
    observation = env.reset()
    calculate_reward=0
    for t in range(250):
        env.render()
        #action = env.action_space.sample()
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        calculate_reward=calculate_reward+reward
        if done:
            break
    if calculate_reward>best_reward:
        best_reward=calculate_reward
        best_param=parameters
print best_reward
print best_param