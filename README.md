# AI gym

reinforcement learning project to train a model to play pinball



## Requirements
- gym
- numpy
- scipy
- skimage
- keras
- matplotlib

## Building the Environment

Once you have all the packages listed above, it is essential to see if the model works. In order to test the environment and make sure we have all the requirements, run the following code
```
python main.py
```
This would spin up a GUI based pinball system that runs on your desktop. ( if you are running on a remote system, remove the following line from ```main.py``` to make the code work.

```
env.render()
```
## How the model works
The mode implements a deeepQ learning model and CNN to train stacks of four images and learns from the visual features.

The report for this repo can be found here: [Reinforcement Learning Report](images/Reinforcement_Learning_Paper.pdf)

## Running the code
To run the code that trains and saves the model, run the following code
```
python pinball_main.py
```
The name of the weights to be saved is to be specified in the code. Incase the code breaks, you can load the code from saved weights and start training from thereon. In order to load weights, you must specify the name of the weights in the following format.
```
model.load_weights('')
```
