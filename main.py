import gym
import numpy as np
import random
env = gym.make('VideoPinball-v0')
best_reward=0
list1=[]
for i_episode in range(2):
    env.render()
    observation = env.reset()
    action=random.randrange(9)
    observation, reward, done, info = env.step(action)
    while not done:
        env.render()
        action=random.randrange(9)
        observation, reward, done, info = env.step(action)
        list1.append(int(reward))
reward=set(list1)

print reward
