import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Concatenate, Reshape, concatenate, LeakyReLU, Add
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras import Sequential, Input
from tensorflow_addons.optimizers import AdamW
from sklearn.utils import shuffle
import matplotlib.pylab as plt
import pickle as pk
import processGames as pg

train_dir = "C:/Users/billy/PycharmProjects/ChessEvaluator/training_data/"
val_dir = "C:/Users/billy/PycharmProjects/ChessEvaluator/val_data/"

def create_residual_block(X_in, filters, regularizerl2):
    X = Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
               data_format="channels_first", kernel_regularizer= regularizerl2)(X_in)
    X = BatchNormalization(axis=1)(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
               data_format="channels_first", kernel_regularizer=regularizerl2)(X)
    X = BatchNormalization(axis=1)(X)
    X = Add()([X, X_in])
    X = Activation("relu")(X)
    return X

def get_full_model_allconv(regularizerl2, filters, residuals):
    X_in = Input(shape=(20, 8, 8))
    X = Conv2D(filters, (3, 3), activation='relu', input_shape=(20, 8, 8), padding='same',
               data_format="channels_first")(X_in)
    X = BatchNormalization(axis = 1)(X)
    X = Activation("relu")(X)
    for i in range(residuals):
        X = create_residual_block(X, filters, regularizerl2)
    X = Conv2D(6, (1, 1), activation='relu', input_shape=(20, 8, 8), padding='same',
           data_format="channels_first")(X)
    X = BatchNormalization(axis = 1)(X)
    X = Flatten()(X)
    X = Activation("relu")(X)
    output = Dense(3, activation = 'softmax')(X)
    return tf.keras.Model(inputs = X_in, outputs = output)

if __name__ == "__main__":

    batchsize= 1024
    training_gen = pg.DataGenerator(train_dir, 92284, batchsize, 3000, 1000, 3000)
    val_gen = pg.DataGenerator(val_dir, 10253, batchsize, 3000, 1000, 3000)

    # regularizerl2 = L2(l2 = 1e-6)
    # model = get_full_model_allconv(regularizerl2, 108, 9)
    model = tf.keras.models.load_model('newest_bigger_model.h5')
    model.summary()

    regAdam = tf.optimizers.Adam(learning_rate=.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False,
        name='Adam')
    stoch = tf.optimizers.SGD(learning_rate= .00008)
    model.compile(optimizer = regAdam, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    history = model.fit(training_gen, validation_data = val_gen,
                        validation_batch_size=batchsize, validation_steps= 10253,  epochs = 1, batch_size= batchsize, verbose=1, steps_per_epoch= 92284, use_multiprocessing= True, workers = 8)
    model.save('newest_bigger_model2.h5')