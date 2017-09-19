"""
Developed by : Gautam Somappa
This is the main python code that trains a model to learn pinball and later on work by itself.

Best worked with a GPU
"""
# Import all packages

import gym
import numpy as np
import scipy
import skimage
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from collections import deque
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#Create the environment

import numpy as np
import random, json, string
from keras.layers import Input,Dense,Conv2D,Flatten
from keras.activations import relu,linear
from keras.models import Sequential
import keras.preprocessing.image
from keras.optimizers import SGD,Adam,RMSprop
from keras.callbacks import ModelCheckpoint

from keras.callbacks import History 
history = History()



class DQNAgent:
    def __init__(self):
        self.epsilon = 0.05
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 10000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        self.batch_size=32
        self.train_start=1000
        self.discount_factor=0.99
        self.length=400000
        
    def get_action(self,stacked_image):
        if random.random() <=self.epsilon:
                print "Random"
                action=random.randrange(9)
        else:
            q = model.predict(stacked_image) 
            print "Calculated"      
            max_Q = np.argmax(q)
            action = max_Q
        return action

    def replay_memory(self, history, action, reward, history1, done):
        memory.append((history, action, reward, history1, done))
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

    
    def train_replay(self):
        if len(memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(memory))
        mini_batch = random.sample(memory, batch_size)

        update_input = np.zeros((batch_size, 84, 84, 4))
        update_target = np.zeros((batch_size, 9))

        for i in range(batch_size):
            history, action, reward, history1, done = mini_batch[i]
            target = model.predict(history)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount_factor * np.amax(target_model.predict(history1)[0])
            update_target[i] = target
            update_input[i] = history
        
        filepath="saved.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss',save_weights_only=True ,save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(update_input, update_target, batch_size=batch_size, epochs=100,callbacks=callbacks_list,verbose=False)


    def update_target_model(self):
        target_model.set_weights(model.get_weights())


def img2array(state):
    state=rgb2gray(state)
    state=resize(state,(84,84))
    state = rescale_intensity(state, out_range=(0, 255))
    state = state.reshape(1, state.shape[0], state.shape[1],1)
    return state

def buildmodel():
    model = Sequential()
    model.add(Conv2D(32, (8, 8), input_shape=(84, 84, 4), activation='relu', strides=(4, 4),
                        kernel_initializer='glorot_uniform'))
    model.add(Conv2D(64, (4, 4), activation='relu', strides=(2, 2),
                        kernel_initializer='glorot_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                        kernel_initializer='glorot_uniform'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(9))
    model.summary()
    model.compile(loss='mse', optimizer=RMSprop(
        lr=0.025, rho=0.95, epsilon=0.01))
    return model
model=buildmodel()
target_model=buildmodel()
model.load_weights('')
num_episodes=1000
#steps=10
memory = deque(maxlen=400000)
env = gym.make('VideoPinball-v0')
env.reset()
stacked_image=np.empty(shape=(1,84,84,4))
agent = DQNAgent()
scores, episodes, global_step = [], [], 0

for e in range(num_episodes):
    done = False
    dead = False
    score=0
    start_live=5
    state = env.reset()
    state = img2array(state)
    stacked_image = np.append(state, stacked_image[:, :, :, :3], axis=3)
    while not done:
        #env.render()
        old_state=stacked_image
        action = agent.get_action(old_state)
        # print action
        next_state, reward, done, info = env.step(action)
        next_state = img2array(next_state)
        stacked_image = np.append(next_state, stacked_image[:, :, :, :3], axis=3)
        agent.replay_memory(old_state, action, reward, stacked_image, done)
        if start_live > info['ale.lives']:
                dead = True
                start_live = info['ale.lives']
        state = agent.train_replay()
        score += reward

        if dead:
            old_state = np.stack((next_state, next_state, next_state, next_state), axis=2)
            old_state = np.reshape([old_state], (1, 84, 84, 4))
            dead = False
        else:
            old_state = stacked_image

        agent.update_target_model()

        if done:
            env.reset()
            scores.append(score)
            episodes.append(e)
            plt.plot(episodes, scores, 'b')
            plt.savefig("Pinball_DQN.png")
            print("episode:", e, "  score:", score, "  memory length:", len(memory),
                  "  epsilon:", agent.epsilon)
    model.save_weights('saved.hdf5')
