from keras import layers, models, optimizers
from keras import backend as K
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.utils import plot_model

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):

    left_input = layers.Input(shape=input_shape)
    right_input = layers.Input(shape=input_shape)

    input = layers.Input(shape=input_shape)
    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(input)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    tunnel_l = models.Model(input, out_caps)
    tunnel_r = models.Model(input, out_caps)

    encoded_l = tunnel_l(left_input)
    encoded_r = tunnel_r(right_input)

    L1_layer = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = layers.Dense(1, activation='sigmoid')(L1_distance)

    train_model = models.Model(inputs=[left_input, right_input], outputs=prediction)

    return train_model


# model = CapsNet(input_shape=[105, 105, 1], n_class=10, routings=3)
# model.summary()
# plot_model(model, to_file='model_capsnet.png')
