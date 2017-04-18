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
from keras.layers import Input,Dense,Conv2D,Flatten
from keras.activations import relu,linear
from keras.models import Sequential
import keras.preprocessing.image
from keras.optimizers import SGD 
import keras.callbacks


##Building the model here

def buildmodel():
    print("Building...")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), activation='relu', input_shape=(84, 84, 4),strides=(4, 4)))
    model.add(Conv2D(64, (4, 4), activation='relu',strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(9,activation='linear'))
    print model.summary()
    sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    print("Model Built")
    return model

model=buildmodel()

# The environment that runs the game


batch=1
epsilon=0.5

def remember(memory,state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
    return memory

env = gym.make('VideoPinball-v0')
env.reset()
stacked_image=np.empty(shape=(1,84,84,4))
memory=[]
rev=[]
for turns in range(2):
    env.render()
    state=env.reset()
    state = rgb2gray(state)
    state = resize(state,(84,84))
    state = rescale_intensity(state, out_range=(0, 255))
    state = state.reshape(1, state.shape[0], state.shape[1],1)
    stacked_image = np.append(state, stacked_image[:, :, :, :3], axis=3)
    for time_t in range(100):
        # turn this on if you want to render
        env.render()
        if random.random() <= epsilon:
            action=env.action_space.sample()
        else:
            q = model.predict(stacked_image)       #input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)
            action_index = max_Q
            action=action_index
        
        print action
        # Advance the game to the next frame based on the action.
        next_state, reward, done, _ = env.step(action)
        old_state=stacked_image
        #next_state = scipy.misc.imresize(next_state, [84,84,3])
        next_state = rgb2gray(next_state)
        next_state = resize(next_state,(84,84))
        next_state = rescale_intensity(next_state, out_range=(0, 255))
        next_state = next_state.reshape(1,next_state.shape[0], next_state.shape[1],1)
        stacked_image = np.append(next_state, stacked_image[:, :, :, :3], axis=3)
        memory.append((old_state, action, reward, stacked_image, done))
        minibatch = random.sample(memory, batch)
        inputs = np.zeros((batch, stacked_image.shape[1], stacked_image.shape[2], stacked_image.shape[3]))   #32, 80, 80, 4
        targets = np.zeros((inputs.shape[0], 9))
        for i in range(0, len(minibatch)):
            state_t = minibatch[i][0]
            action_t = minibatch[i][1]
            reward_t = minibatch[i][2]
            state_t1 = minibatch[i][3]
            terminal = minibatch[i][4]
            inputs[i:i + 1] = state_t
            targets[i] = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            if terminal:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + 0.01* np.max(Q_sa)

            loss = model.train_on_batch(inputs, targets)
        print loss
