import gym
import numpy as np
import model
import scipy
import skimage
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import copy
#Create the environment

import numpy as np
import random, json, string
from keras.layers import Input,Dense,Conv2D
from keras.activations import relu,linear
from keras.models import Sequential
import keras.preprocessing.image
from keras.optimizers import SGD 
import keras.callbacks


##Building the model here

def buildmodel():
    print("Building...")
    model = Sequential()
    img_channels=1
    img_rows=84
    img_cols=84
    model=Sequential()
    model.add(Conv2D(32, (8, 8), activation='relu', input_shape=(84, 84, 4),strides=(4, 4)))
    model.add(Conv2D(64, (4, 4), activation='relu', input_shape=(32, 20, 20),strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 9, 9)))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(8,activation='linear'))
    print model.summary()
    sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    print("Model Built")
    return model

model=buildmodel()

# The environment that runs the game


BATCH=32


def remember(memory,state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
    return memory

env = gym.make('VideoPinball-v0')
env.reset()
stacked_image=np.empty(shape=(1,4,84,84))
memory=[]
for turns in range(2):
    env.render()
    state=env.reset()
    state = rgb2gray(state)
    state = resize(state,(84,84))
    state = rescale_intensity(state, out_range=(0, 255))
    state = state.reshape(1, 1, state.shape[0], state.shape[1])
    stacked_image = np.append(state, stacked_image[:, :3, :, :], axis=1)
    for time_t in range(100):
            # turn this on if you want to render
            env.render()
            action=env.action_space.sample()
            # Advance the game to the next frame based on the action.
            next_state, reward, done, _ = env.step(action)
            old_state=stacked_image
            next_state = scipy.misc.imresize(next_state, [84,84,3])
            next_state = rgb2gray(next_state)
            next_state = resize(next_state,(84,84))
            next_state = rescale_intensity(next_state, out_range=(0, 255))
            next_state = next_state.reshape(1, 1, next_state.shape[0], next_state.shape[1])
            stacked_image = np.append(next_state, stacked_image[:, :3, :, :], axis=1)
            memory.append((old_state, action, reward, stacked_image, done))
            val= [x[0].shape for x in memory]

print val

