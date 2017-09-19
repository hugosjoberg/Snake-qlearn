import snake as game
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import SGD , Adam
#Shape of the image imported into the NN
input_shape=(84,84,3)
#nb_actions is the number of actions the player can do in the game
nb_actions=4
learning_rate=1e-4
#Building a NN model for interpretating the image from the game
#The architecture is borrowed from https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
#First wanted to try the DeepMind Atari model but comments recommended the one below instead
#Performance is supposed to be a lot better.
model = Sequential()
#model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(32, (8, 8), strides=(4, 4),input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

#I chosed to use a Adam optimizer, I have used it before with good results
adam=Adam(lr=learning_rate)
model.compile(loss='mean_squared_error',
            optimizer=adam)
print(model.summary())
