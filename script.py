import sys
import numpy as np
import pandas as pd
# from matplotlib.pyplot import imread
from skimage.io import imread
from skimage.transform import resize
import pickle
import os
import matplotlib.pyplot as plt

import cv2
from time import time
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng
from keras.utils import plot_model
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from time import time
import glob
from IPython.display import Image

train_folder = "data/images_background/"
val_folder = 'data/images_evaluation/'
save_path = 'data/'
train_type = 'capsule' #CONVOLUTIONAL or CAPSULE

def loadimgs(path,n = 0):
    '''
    path => Path of train directory or test directory
    '''
    X=[]
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    # we load every alphabet seperately so we can isolate them later
    for alphabet in [x for x in os.listdir(path) if not x.startswith('.')]:
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y,None]
        alphabet_path = os.path.join(path,alphabet)
        # every letter/category has it's own column in the array, so  load seperately
        for letter in [x for x in os.listdir(alphabet_path) if not x.startswith('.')]:
            cat_dict[curr_y] = (alphabet, letter)
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
            # read all the images in the current category
            for filename in [x for x in os.listdir(letter_path) if not x.startswith('.')]:
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
#                 image = resize(image, (52, 52))
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X,y,lang_dict

def initialize_weights(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)

print("Training alphabets: \n")
print(list(train_classes.keys()))

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)

print("Validation alphabets:", end="\n\n")
print(list(val_classes.keys()))


def get_batch(batch_size, s="train"):
    """Create batch of n pairs, half same class, half different class"""
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size,), replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets

def generate(batch_size, s="train"):
    """a generator for batches, so model.fit_generator can be used. """
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)


def make_oneshot_task(N, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape

    indices = rng.randint(0, n_examples, size=(N,))
    if language is not None:  # if language is specified, select characters for that language
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low, high), size=(N,), replace=False)

    else:  # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes), size=(N,), replace=False)
    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
    test_image = np.asarray([X[true_category, ex1, :, :]] * N).reshape(N, w, h, 1)
    support_set = X[categories, indices, :, :]
    support_set[0, :, :] = X[true_category, ex2]
    support_set = support_set.reshape(N, w, h, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image, support_set]

    return pairs, targets


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net

# Hyper parameters
evaluate_every = 200 # interval for evaluating on one-shot tasks
batch_size = 32
n_iter = 20000 # No. of training iterations
N_way = 20 # how many classes for testing one-shot tasks
n_val = 250 # how many one-shot tasks to validate on
best = -1

model_path = "checkpoints_{0}".format(train_type)

os.makedirs("checkpoints_{0}".format(train_type), exist_ok=True)
file_path = "checkpoints_{0}".format(train_type)+"/siamese-epoch-{epoch:05d}-lr-" + "-train_loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(file_path,
                             monitor=['loss', 'val_loss'],
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='min',
                             period=2)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=0)
lr_decay = LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch))



def get_capsnet_model(input_shape, n_class, routings):

    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    input = Input(shape=input_shape)
    # Layer 1: Just a conventional Conv2D layer
    conv1 = Conv2D(filters=128, kernel_size=36, strides=1, padding='valid', activation='relu', name='conv1',
                  kernel_initializer=initialize_weights, bias_initializer=initialize_bias,
                   kernel_regularizer=l2(2e-4),)(input)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=16, kernel_size=36, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    tunnel = Model(input, out_caps)
    tunnel.summary()

    encoded_l = tunnel(left_input)
    encoded_r = tunnel(right_input)

    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    train_model = Model(inputs=[left_input, right_input], outputs=prediction)

    return train_model

ckpts = glob.glob("checkpoints_{0}".format(train_type)+"/*.hdf5")
initial_epoch = 0
if len(ckpts) != 0 and False:
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print("loading from checkpoint: ", latest_ckpt)
    initial_epoch = int(latest_ckpt[latest_ckpt.find("-epoch-") + len("-epoch-"):latest_ckpt.rfind("-lr-")])
    model = load_model(latest_ckpt, custom_objects={'CapsuleLayer': CapsuleLayer, 'Length': Length, 'PrimaryCap':PrimaryCap })
else:
    if train_type == 'capsule':
        print("***New model for capsule network loaded***")
        model = get_capsnet_model(input_shape=[105, 105, 1], n_class=10, routings=3)
    elif train_type == 'convolutional':
        print("***New model for convolutional network loaded***")
        model = get_siamese_model((105, 105, 1))
model.summary()
plot_model(model, to_file='model.png')
Image(retina=True, filename='model.png')

optimizer = Adam(lr = 0.001)
model.compile(loss="binary_crossentropy",optimizer=optimizer)

model.fit_generator(generator=generate(32),
                    steps_per_epoch=100,
                    epochs=1000,
                    initial_epoch=initial_epoch,
                    validation_data=generate(32, 'val'),
                    validation_steps=2,
                    callbacks=[checkpoint, tensorboard, lr_decay],
                    use_multiprocessing=True)
