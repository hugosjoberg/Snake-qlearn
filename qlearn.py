import snake as game
import argparse
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import SGD , Adam
import numpy as np
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from collections import deque
import random
import json


INPUT_SHAPE=(80,80,4) #Shape of the image imported into the NN
NB_ACTIONS = 4 #NB_ACTIONS is the number of actions the player can do in the game
NB_FRAMES = 1 #NB_FRAMES is the number of frames that are presented when an action can be performed
OBSERVATION = 3200
REPLAY_MEMORY = 50000
BATCH = 32
GAME_INPUT = [0,1,2,3]
EXPLORE = 3000000
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.0001
LEARNING_RATE = 1e-4
GAMMA = 0.99 # decay rate of past observations


#Building a NN model for interpretating the image from the game
#The architecture is borrowed from https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
#First wanted to try the DeepMind Atari model but comments recommended the one below instead
#Performance is supposed to be a lot better.

def build_model():
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4),input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(NB_ACTIONS))
    model.add(Activation('linear'))

    #I chosed to use a Adam optimizer, I have used it before with good results
    adam=Adam(lr=LEARNING_RATE)
    model.compile(loss='mean_squared_error',
                    optimizer=adam)
    print(model.summary())
    return model

def train_network(model,args):

    game_state = game.Game() #Starting up a game
    game_state.set_start_state()
    game_image,score,game_lost = game_state.run(4) #The game is started but no action is performed
    D = deque()

    #Make image black and white
    x_t = skimage.color.rgb2gray(game_image)
    #Resize the image to 80x80 pixels
    x_t = skimage.transform.resize(x_t,(80,80))
    #Change the intensity of colors, maximizing the intensities.
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    # Stacking 4 images for the agent to get understanding of speed
    s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
    # Reshape to make keras like it
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    terminal = False
    t = 0
    while(True):

        loss = 0
        Q_sa = 0
        action_index = 4
        r_t = 0
        a_t = 'no nothing'
        if terminal:
            game_state.set_start_state()
        if t % NB_FRAMES == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(NB_ACTIONS)
                a_t = GAME_INPUT[action_index]
            else:
                action_index = np.argmax(model.predict(s_t))
                a_t = GAME_INPUT[action_index]

        #Reduce learning rate over time
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.run(a_t)

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if(len(D) > REPLAY_MEMORY):
            D.popleft()
        if(t>OBSERVE):

            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], NB_ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)
                #action_index = np.argmax(model.predict(s_t))

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if t % 100 == 0:

            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", LEARNING_RATE, "/ ACTION", action_index, "/ REWARD", r_t, \
                "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

def playGame(args):
        model = build_model()
        train_network(model,args)

def main():
    parser = argparse.ArgumentParser(description='How you would like your program to run')
    parser.add_argument('-m','--mode',help='Train / Run', required =True)
    args = vars(parser.parse_args())
    playGame(args)


if __name__ == "__main__":
    main()
