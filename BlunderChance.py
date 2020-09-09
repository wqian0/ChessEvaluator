import codecs
import gc
import threading

import pandas as pd
import numpy as np
import scipy as sp
from scipy import special
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Concatenate, Reshape, concatenate, LeakyReLU, Add
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras import Sequential, Input
from tensorflow_addons.optimizers import AdamW
from sklearn.utils import shuffle
import matplotlib.pylab as plt
import pickle as pk
import chess
import chess.pgn
import chess.engine

# pgns = open("lichess_db_standard_rated_2017-03.pgn")
training_data = pd.read_csv("training_data_inc.csv")
val_data = pd.read_csv("val_data_inc.csv")

train_dir = "C:/Users/billy/PycharmProjects/ChessEvaluator/training_data/"
val_dir = "C:/Users/billy/PycharmProjects/ChessEvaluator/val_data/"

# train_filename = "training_data_inc.csv"
# val_filename = "val_data_inc.csv"

fen_all_pieces = 'KQRBNPkqrbnp'
indices = {}
for i in range(len(fen_all_pieces)):
    indices[fen_all_pieces[i]] = i

def convert_board_part_compressed(input): # -1, 0, 1 encoding
    output = np.zeros((6, 8, 8), dtype=np.float16)
    modified_input = input
    for i in range(9):
        modified_input = modified_input.replace(str(i), '*' * i)
    modified_input = modified_input.replace("/", "")
    for i in range(8):
        for j in range(8):
            currLoc = 8*i + j
            letter = modified_input[currLoc]
            if letter.isalpha():
                index = indices[modified_input[currLoc]] % 6
                if letter.isupper():
                    output[index][i][j] = 1
                else:
                    output[index][i][j] = -1
    return output

def convert_board_part(input): # One hot encoding for first 12 channels
    output = np.zeros((19, 8, 8), dtype = np.float16)
    modified_input = input
    for i in range(9):
        modified_input = modified_input.replace(str(i), '*'*i)
    modified_input = modified_input.replace("/", "")
    for i in range(8):
        for j in range(8):
            currLoc = 8*i + j
            if modified_input[currLoc].isalpha():
                output[indices[modified_input[currLoc]]][i][j] = 1
    return output

#3q2k1/p4ppp/8/2pn4/8/8/P2n1P1P/3R1R1K w - - 0 25
def convert_board_part_fast(input): # One hot encoding for first 12 channels
    output = np.zeros((19, 8, 8), dtype = np.float16)
    modified_input = input
    row = 0
    col = 0
    for i in range(len(input)):
        if input[i].isdigit():
            col += int(input[i])
        elif input[i].isalpha():
            output[indices[input[i]]][row][col] = 1
            col += 1
        else:
            row += 1
            col = 0
    return output

def create_residual_block(X_in, regularizerl2):
    X = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
               data_format="channels_first", kernel_regularizer= regularizerl2)(X_in)
    X = BatchNormalization(axis=1)(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
               data_format="channels_first", kernel_regularizer=regularizerl2)(X)
    X = BatchNormalization(axis=1)(X)
    X = Add()([X, X_in])
    X = Activation("relu")(X)
    return X

def get_full_model(regularizerl2, residuals):
    X_in = Input(shape=(6, 8, 8))
    X_extra = Input(shape=(7,))
    X = Conv2D(64, (3, 3), activation='relu', input_shape=(6, 8, 8), padding='same',
               data_format="channels_first", kernel_regularizer=regularizerl2)(X_in)
    X = BatchNormalization(axis = 1)(X)
    X = Activation("relu")(X)
    for i in range(residuals):
        X = create_residual_block(X, regularizerl2)
    X = Conv2D(8, (1, 1), activation='relu', input_shape=(6, 8, 8), padding='same',
           data_format="channels_first", kernel_regularizer=regularizerl2)(X)
    X = BatchNormalization(axis = 1)(X)
    X = Activation("relu")(X)
    X = Flatten()(X)
    X = Concatenate()([X, X_extra])
    X = Dense(256, kernel_regularizer= regularizerl2)(X)
    X = Activation("relu")(X)
    output = Dense(3, kernel_regularizer= regularizerl2, activation = 'softmax')(X)
    return tf.keras.Model(inputs = [X_in, X_extra], outputs = output)

def get_big_model(regularizerl2, residuals):
    X_in = Input(shape=(19, 8, 8))
    X = Conv2D(64, (3, 3), activation='relu', input_shape=(19, 8, 8), padding='same',
               data_format="channels_first", kernel_regularizer=regularizerl2)(X_in)
    X = BatchNormalization(axis = 1)(X)
    X = Activation("relu")(X)
    for i in range(residuals):
        X = create_residual_block(X, regularizerl2)
    X = Conv2D(2, (1, 1), activation='relu', input_shape=(19, 8, 8), padding='same',
           data_format="channels_first", kernel_regularizer=regularizerl2)(X)
    X = BatchNormalization(axis = 1)(X)
    X = Activation("relu")(X)
    X = Flatten()(X)
    output = Dense(3, kernel_regularizer= regularizerl2, activation = 'softmax')(X)
    return tf.keras.Model(inputs = X_in, outputs = output)

def fully_connected_part(regularizerl2):
    model = Sequential()
    model.add(Dense(512, activation='relu', kernel_regularizer = regularizerl2))
    model.add(Dense(256, activation='relu', kernel_regularizer = regularizerl2))
    model.add(Dense(3, activation='softmax', kernel_regularizer = regularizerl2))
    return model

def get_conv_model(regularizerl2):
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), activation='relu', input_shape=(6, 8, 8), padding='same', data_format="channels_first", kernel_regularizer=regularizerl2))
    model.add(BatchNormalization(axis = 1))
    model.add(
        Conv2D(64, (3, 3), activation='relu', input_shape=(6, 8, 8), padding='same', data_format="channels_first", kernel_regularizer=regularizerl2))
    model.add(BatchNormalization(axis = 1))
    model.add(
        Conv2D(64, (3, 3), activation='relu', input_shape=(6, 8, 8), padding='same', data_format="channels_first", kernel_regularizer=regularizerl2))
    model.add(BatchNormalization(axis = 1))
    model.add(
        Conv2D(64, (3, 3), activation='relu', input_shape=(6, 8, 8), padding='same', data_format="channels_first", kernel_regularizer=regularizerl2))
    model.add(BatchNormalization(axis = 1))
    return model

def get_combined_model():
    input1 = Input(shape = (6, 8, 8))
    input2 = Input(shape = (7,))

    regularizerl2 = L2(l2 = 1e-6)
    conv_model = get_conv_model(regularizerl2)
    fc_model = fully_connected_part(regularizerl2)
    cnn_output = Flatten()(conv_model(input1))
    output = fc_model(Concatenate(axis = -1)([cnn_output, input2]))
    model = tf.keras.models.Model(inputs = [input1, input2], outputs = output)
    model.summary()
    return model

def convert_fen(fen, include_enpassant = False):
    parts = fen.split(" ")
    board = convert_board_part(parts[0])
    flattened_board = np.ndarray.flatten(board)
    col = np.zeros(1, dtype = np.float16)
    if parts[1] == 'w':
        col[0] = 1
    else:
        col[0] = 0
    castling = np.zeros(4, dtype = np.float16)
    castling_rights = parts[2]
    if "K" in castling_rights:
        castling[0] = 1
    if "Q" in castling_rights:
        castling[1] = 1
    if "k" in castling_rights:
        castling[2] = 1
    if "q" in castling_rights:
        castling[3] = 1
    else:
        return np.concatenate((col, flattened_board, castling))
def convert_fen2(fen, compressed = False, flip = False):
    parts = fen.split(" ")
    if compressed:
        board = convert_board_part_compressed(parts[0])
    else:
        board = convert_board_part(parts[0])
    if flip:
        board = -np.flip(board, axis = 1)
    col_and_castling = np.zeros(5)
    if parts[1] == 'w':
        col_and_castling[0] = 1
    elif compressed:
        col_and_castling[0] = -1
    else:
        col_and_castling[0] = 0
    castling_rights = parts[2]
    if "K" in castling_rights:
        col_and_castling[1] = 1
    if "Q" in castling_rights:
        col_and_castling[2] = 1
    if "k" in castling_rights:
        col_and_castling[3] = -1 * compressed
    if "q" in castling_rights:
        col_and_castling[4] = -1 * compressed
    if flip:
        col_and_castling *= -1
    return [board, col_and_castling]

def combine_fen_rating(fen, white_elo, black_elo):
    parts = fen.split(" ")
    board = convert_board_part(parts[0])
    if parts[1] == 'w':
        board[12] += 1
    castling_rights = parts[2]
    if "K" in castling_rights:
        board[13] += 1
    if "Q" in castling_rights:
        board[14] += 1
    if "k" in castling_rights:
        board[15] += 1
    if "q" in castling_rights:
        board[16] += 1
    board[17] += 1
    board[18] += 1
    board[17] *= white_elo
    board[18] *= black_elo
    return board

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dir, num_inputs, batch_size, shuffle=True):
        'Initialization'
        self.dir = dir
        self.num_inputs = num_inputs
        self.batch_size = batch_size
        self.indices = np.arange(num_inputs)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return self.num_inputs // self.batch_size

    def __getitem__(self, index):
        X, y = self.__data_generation(self.indices[index])
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, index):
        boards = np.load(self.dir+'board_'+str(index)+'.npy')
        extras = np.load(self.dir+'extra_'+str(index)+'.npy')
        outcomes = np.load(self.dir+'outcome_'+str(index)+'.npy')
        return [boards, extra], outcomes


def generate_from_csv(csv, batchsize, chunksize, elo_norm):
    df_it = pd.read_csv(csv, iterator = True)
    while True:
        board_parts = []
        extra_parts = []
        try:
            df = df_it.get_chunk(chunksize)
        except StopIteration:
            df_it = pd.read_csv(csv, iterator=True)
            df = df_it.get_chunk(chunksize)
        indices = df.index.tolist()
        np.random.shuffle(indices)
        results = np.asarray(df['Result'][indices[:batchsize]])
        for i in range(batchsize):
            board_part, extra_part = convert_fen2(df['FEN'][indices[i]], compressed= True)
            board_parts.append(board_part)
            elos = np.array([df['WhiteElo'][indices[i]], df['BlackElo'][indices[i]]]) / elo_norm
            extra_part = np.concatenate([extra_part, elos])
            extra_parts.append(extra_part)
        yield [np.asarray(board_parts), np.asarray(extra_parts)], results

def generate_from_csv_combined(csv, batchsize, chunksize, elo_norm):
    df_it = pd.read_csv(csv, iterator = True)
    while True:
        inputs = np.zeros((batchsize, 19, 8, 8))
        try:
            df = df_it.get_chunk(chunksize)
        except StopIteration:
            df_it = pd.read_csv(csv, iterator=True)
            df = df_it.get_chunk(chunksize)
        indices = df.index.tolist()
        np.random.shuffle(indices)
        results = np.asarray(df['Result'][indices[:batchsize]])
        for i in range(batchsize):
            input = combine_fen_rating(df['FEN'][indices[i]], df['WhiteElo'][indices[i]]/ elo_norm, df['BlackElo'][indices[i]]/ elo_norm)
            inputs[i] = input
        yield inputs, results


##Generates randomly selected training/validation data directly from a csv/dataframe. For greater training speedup, pre-process the FENs
def generate_from_df(df, batchsize, elo_norm):
    while True:
        board_parts = []
        extra_parts = []
        indices = np.random.choice(len(df), batchsize)
        results = np.asarray(df['Result'][indices])
        for i in range(len(indices)):
            board_part, extra_part = convert_fen2(df['FEN'][indices[i]], compressed= True)
            board_parts.append(board_part)
            elos = np.array([df['WhiteElo'][indices[i]], df['BlackElo'][indices[i]]]) / elo_norm
            extra_part = np.concatenate([extra_part, elos])
            extra_parts.append(extra_part)
        yield [np.asarray(board_parts), np.asarray(extra_parts)], results

def generate_from_df_combined(df, batchsize, elo_norm):
    while True:
        inputs = np.zeros((batchsize, 19, 8, 8))
        indices = np.random.choice(len(df), batchsize)
        results = np.asarray(df['Result'][indices])
        for i in range(len(indices)):
            input = combine_fen_rating(df['FEN'][indices[i]], df['WhiteElo'][indices[i]]/ elo_norm, df['BlackElo'][indices[i]]/ elo_norm)
            inputs[i] = input
        yield inputs, results

def read_all_games(pgnList, positions_cap):
    FENs = []
    elos = []
    results = [] # 0, white wins, 1 draw, 2 black wins
    curr = chess.pgn.read_game(pgnList)
    gameNo = 0
    # while gameNo < 1065000:
    #     print(gameNo)
    #     curr = chess.pgn.read_game(pgnList)
    #     gameNo += 1
    while curr is not None and len(elos) < positions_cap:
        board = curr.board()
        moves = list(curr.mainline_moves())
        result = 0
        if curr.headers['Result'] == '1/2-1/2':
            result = 1
        if curr.headers['Result'] == '0-1':
            result = 2
        plusIndex = curr.headers['TimeControl'].find('+')
        # filters out time-controls below 3-minute blitz, and games with less than 2 moves
        if len(moves) > 1 and plusIndex != -1 and int(curr.headers['TimeControl'][0:plusIndex]) >= 180 and int(curr.headers['TimeControl'][plusIndex + 1 :]) >= 2:
            whiteElo = curr.headers['WhiteElo']
            blackElo = curr.headers['BlackElo']
            print(gameNo, len(elos), result, whiteElo, blackElo, curr.headers['TimeControl'])
            for i in range(len(moves)):
                FENs.append(board.fen())
                board.push(moves[i])
            FENs.append(board.fen())
            elos.extend([[whiteElo, blackElo] for j in range(len(moves) + 1)])
            results.extend([result for j in range(len(moves) + 1)])
        curr = chess.pgn.read_game(pgnList)
        gameNo += 1
    return FENs, elos, results
# def get_dataset(file_path, batch_size):
#   dataset = tf.data.experimental.make_csv_dataset(
#       file_path,
#       batch_size=batch_size,
#       select_columns= [1,2,3,4],
#       label_name='Result',
#       num_epochs = 1)
#   return dataset
#
# def combine_fen_rating_tf(fens, white_elo, black_elo):
#     boards = []
#     for i in range(len(fens)):
#         fen = fens[i].numpy().decode('utf-8')
#         parts = fen.split(" ")
#         board = convert_board_part(parts[0])
#         if parts[1] == 'w':
#             board[12] += 1
#         castling_rights = parts[2]
#         if "K" in castling_rights:
#             board[13] += 1
#         if "Q" in castling_rights:
#             board[14] += 1
#         if "k" in castling_rights:
#             board[15] += 1
#         if "q" in castling_rights:
#             board[16] += 1
#         board[17] += 1
#         board[18] += 1
#         board[17] *= white_elo[i]
#         board[18] *= black_elo[i]
#         boards.append(board)
#     return np.array(boards)
if __name__ == "__main__":
    board_parts = []
    extra_parts = []
    outcomes = []
    for i in range(len(training_data) // 1024):
        print(i)
        for j in range(1024):
            board, extra = convert_fen2(training_data['FEN'][i * 1024 + j], compressed = True)
            board_parts.append(board)
            extra_parts.append(extra)
            outcomes.append(training_data['Result'][i * 1024 + j])
        board_parts = np.array(board_parts)
        extra_parts = np.array(extra_parts)
        outcomes = np.array(outcomes)
        np.save(open(train_dir +'board_' + str(i) + ".npy", 'wb'), board_parts)
        np.save(open(train_dir + 'extra_' + str(i) + ".npy", 'wb'), extra_parts)
        np.save(open(train_dir+'outcome_'+ str(i)+".npy", 'wb'), outcomes)
        board_parts = []
        extra_parts = []
        outcomes = []

    board_parts = []
    extra_parts = []
    outcomes = []
    for i in range(len(val_data) // 1024):
        print(i)
        for j in range(1024):
            board, extra = convert_fen2(val_data['FEN'][i * 1024 + j], compressed = True)
            board_parts.append(board)
            extra_parts.append(extra)
            outcomes.append(val_data['Result'][i * 1024 + j])
        board_parts = np.array(board_parts)
        extra_parts = np.array(extra_parts)
        outcomes = np.array(outcomes)
        np.save(open(val_dir +'board_' + str(i) + ".npy", 'wb'), board_parts)
        np.save(open(val_dir + 'extra_' + str(i) + ".npy", 'wb'), extra_parts)
        np.save(open(val_dir+'outcome_'+ str(i)+".npy", 'wb'), outcomes)
        board_parts = []
        extra_parts = []
        outcomes = []

    # # batchsize= 1024
    # # training_data = get_dataset(train_filename, batchsize)
    # # val_data = get_dataset(val_filename, batchsize)
    # # training_data = training_data.map(lambda x, y: (tf.py_function(func = combine_fen_rating_tf, inp = [x['FEN'], x['WhiteElo'], x['BlackElo']], Tout = tf.float16), y), num_parallel_calls= 8)
    # # val_data = val_data.map(lambda x, y: (tf.py_function(func = combine_fen_rating_tf, inp = [x['FEN'], x['WhiteElo'], x['BlackElo']], Tout = tf.float16), y), num_parallel_calls= 8)
    # regularizerl2 = L2(l2 = 1e-5)
    # #model = get_big_model(regularizerl2, 1)
    # model = get_combined_model()
    # model.summary()
    # decayedAdam = AdamW(weight_decay = 1e-6, learning_rate = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='AdamW')
    # regAdam = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    #     name='Adam')
    # stoch = tf.keras.optimizers.SGD(learning_rate= .001, momentum = .8, nesterov= True)
    # model.compile(optimizer = regAdam, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    # history = model.fit(generate_from_df(training_data, batchsize, 3000), validation_data = generate_from_df(val_data, batchsize , 3000),
    #                     validation_batch_size=batchsize, validation_steps=7500000// (batchsize * 16),  epochs = 10, batch_size= batchsize, verbose=1, steps_per_epoch= 60000000 // (batchsize * 16))
    # model.save_weights('newest2.h5')
    # plt.plot(history.history['sparse_categorical_accuracy'])
    # plt.plot(history.history['val_sparse_categorical_accuracy'])
    # plt.show()

