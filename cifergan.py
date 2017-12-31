# coding:utf-8

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.datasets import mnist
import dutil
from PIL import Image
from keras.datasets import cifar10

def generator():
    model = Sequential()

    model.add(Dense(units=8 * 8 * 128, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2, 2)))
    model.add(Activation('tanh'))

    return model

generator().summary()


def discriminator():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), input_shape=(32, 32, 3)))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2)))
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

discriminator().summary()


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    X_train = (X_train - 127.5)/127.5
    x_train = []
    for i, a in enumerate(y_train):
        if a == 9:
            x_train.append(X_train[i])
    X_train = np.array(x_train)
    print("X_train.shape", X_train.shape)
    
    d = discriminator()
    # d = load_model("d.h5")
    d.compile(optimizer=Adam(lr=1e-5, beta_1=0.1), loss='binary_crossentropy')
    
    g = generator()
    # g = load_model("g.h5")
    d.trainable = False
    
    dcgan = Sequential([g, d])
    dcgan.compile(optimizer=Adam(lr=2e-4, beta_1=0.5), loss='binary_crossentropy')
    
    EPOCH_SIZE = 100000
    BATCH_SIZE = 32
    Z_DIM = 100
    
    for epoch in range(1, EPOCH_SIZE + 1):
        #print(f"epoch {epoch}")
        gx = g.predict(np.random.uniform(-1, 1, size=[1, Z_DIM]), verbose=0)
        print(d.predict(gx)[0, 0])
        plt.imshow(gx.reshape(32, 32, 3) * 127.5 + 127.5)
        plt.savefig("./images/" + str(epoch) + '.png')
        plt.show()
        print("epo" ,epoch)
    
    
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            gx = g.predict(np.random.uniform(-1, 1, size=[BATCH_SIZE, Z_DIM]), verbose=0)
    
            d_x = np.concatenate([X_train[index * BATCH_SIZE: (index + 1) * BATCH_SIZE].reshape(BATCH_SIZE, 32, 32, 3), gx])
            d_y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
    
            d_loss = d.train_on_batch(d_x, d_y)
    
            g_loss = dcgan.train_on_batch(np.random.uniform(-1, 1, size=[BATCH_SIZE, Z_DIM]), [1] * BATCH_SIZE)
    
        d.save("d.h5")
        g.save("g.h5")
    
    
