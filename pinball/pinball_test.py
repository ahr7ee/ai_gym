import gym
import numpy as np
env = gym.make('VideoPinball-ram-v0')
env.reset()
for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    #print "Observation:",observation
    if reward>1:
        print "Reward: ",reward
        print "Observation:",observation

