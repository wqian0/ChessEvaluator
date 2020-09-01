import codecs

import pandas as pd
import numpy as np
import scipy as sp
from scipy import special
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Concatenate
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras import Sequential, Input
from sklearn.utils import shuffle
import matplotlib.pylab as plt
import pickle as pk
import chess
import chess.pgn
import chess.engine

file = open("chessData.csv", "r")
chess_data = pd.read_csv(file, encoding = 'utf_8_sig')

tactics_f = open("tactics.pgn", "r")

fen_all_pieces = 'KQRBNPkqrbnp'
indices = {}
for i in range(len(fen_all_pieces)):
    indices[fen_all_pieces[i]] = i



def convert_board_part(input):
    output = np.zeros((12, 8, 8), dtype = np.float16)
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

def get_best_move(fen, model):
    board = chess.Board(fen)
    legal = board.legal_moves
    moves = []
    index = 0
    scores = np.zeros(legal.count())
    for move in legal:
        moves.append(move)
        temp_board = board.copy()
        temp_board.push(move)
        curr_score = model.predict(np.array([convert_fen(temp_board.fen())]))[0][0]
        scores[index] = curr_score
        index += 1
    if board.turn == chess.WHITE:
        bestIndex = np.argmax(scores)
    else:
        bestIndex = np.argmin(scores)
    print(moves, scores)
    return moves[bestIndex]



#capping scores
def clean_evals(input, soft_cap, hard_cap):
    output = np.zeros(len(input), dtype = np.float16)
    for i in range(len(input)):
        if input[i][0] == '#':
            sign = 1
            if input[i][1] == '-':
                sign = -1
            mate_distance = int(input[i][2:])
            output[i] = sign * (soft_cap + (22 - mate_distance) * (hard_cap - soft_cap) / 22)
            # if input[i][2] == '0':
            #     output[i] = sign * hard_cap
            # else:
            #     output[i] = sign * soft_cap
        else:
            output[i] = float(input[i]) / 100
        print(i, output[i])
    return np.clip(output, a_min = -hard_cap, a_max = hard_cap)

def sigmoid(X):
    return 1/(1+np.exp(-X))
def inv_sigmoid(Y):
    return np.log(Y/(1-Y))

def convert_string_coordinate(input):
    file = input[0]
    rank = int(input[1]) - 1
    col = ord(file) - 97
    return (rank, col)

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
    if include_enpassant:
        en_passant_rights = parts[3]
        en_passant = np.zeros((8, 8), dtype = np.float16)
        if len(en_passant_rights) > 1:
            rank, column = convert_string_coordinate(en_passant_rights)
            en_passant[rank][column] = 1
        en_passant_flattened = np.ndarray.flatten(en_passant)
        return np.concatenate((col, flattened_board, castling, en_passant_flattened))
    else:
        return np.concatenate((col, flattened_board, castling))

def convert_fen2(fen):
    parts = fen.split(" ")
    board = convert_board_part(parts[0])
    col_and_castling = np.zeros(5)
    if parts[1] == 'w':
        col_and_castling[0] = 1
    else:
        col_and_castling[0] = 0
    castling_rights = parts[2]
    if "K" in castling_rights:
        col_and_castling[1] = 1
    if "Q" in castling_rights:
        col_and_castling[2] = 1
    if "k" in castling_rights:
        col_and_castling[3] = 1
    if "q" in castling_rights:
        col_and_castling[4] = 1
    return [board, col_and_castling]
def get_classifications_full(evals, b_win_score, w_win_score):
    white_wins = np.zeros(3)
    even = np.zeros(3)
    black_wins = np.zeros(3)
    white_wins[0] = 1
    even[1] = 1
    black_wins[2] = 1
    classifications = []
    for e in evals:
        if e < b_win_score:
            classifications.append(black_wins)
        elif b_win_score <= e and e <= w_win_score:
            classifications.append(even)
        else:
            classifications.append(white_wins)
    return classifications

def get_classifications(evals, b_win_score, w_win_score):
    classifications = []
    for e in evals:
        if e < b_win_score:
            classifications.append(2)
        elif b_win_score <= e and e <= w_win_score:
            classifications.append(1)
        else:
            classifications.append(0)
    return classifications
def get_classification_model():
    model = Sequential()
    model.add(Dense(1048, input_dim= 773))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(500, input_dim=773))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(50, input_dim=773))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    return model


def get_regression_model():
    model = Sequential()
    model.add(Dense(1500, input_dim=773))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Dropout(0.2))


    model.add(Dense(1500))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.2))


    model.add(Dense(1500))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.summary()
    return model

def get_conv_model(regularizerl2):
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), activation='relu', input_shape=(12, 8, 8), padding='same', kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last",
                     kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(
        Conv2D(64, (3, 3), activation='relu', input_shape=(12, 8, 8), padding='same', kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(
        Conv2D(64, (3, 3), activation='relu', input_shape=(12, 8, 8), padding='same', kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(
        Conv2D(64, (3, 3), activation='relu', input_shape=(12, 8, 8), padding='same', kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(Flatten())
    return model

def fully_connected_part(initializer, regularizerl2):
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizerl2))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizerl2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


def train_classification(model, train_x, train_y, val_data):
    return model.fit(train_x, train_y, epochs = 5, verbose = 1, validation_data = val_data, batch_size = 128)

def train_regression(model, train_x, train_y, val_data, weights = None):
    return model.fit(train_x, train_y, epochs = 5, verbose = 1, validation_data = val_data, use_multiprocessing= True, sample_weight= weights, batch_size= 128)

capped_evals = pk.load(open("capped_evals_148_255_gradated.pickle", "rb"))

# plt.hist(capped_evals, bins = 50)
# plt.show()
# min_val = np.min(capped_evals)
# max_val = np.max(capped_evals)
# #capped_evals = (capped_evals - min_val)/ (max_val - min_val)
# mean = np.mean(capped_evals)
# std = np.std(capped_evals)
# capped_evals = (capped_evals - mean) / std
# weights = np.array([.8 + .4 * abs(x - mean) for x in capped_evals])

converted_fens = []

board_parts = []
extra_parts = []


# for i in range(9):
#     # converted_fens.extend(pk.load(open("converted_fens_773_"+str(i)+".pickle", "rb")))
#     board_parts.extend(pk.load(open("board_parts_"+str(i)+".pickle", "rb")))
#     extra_parts.extend(pk.load(open("extra_parts_"+str(i)+".pickle", "rb")))

for i in range(len(chess_data['FEN'])):
    board_part, extra = convert_fen2(chess_data['FEN'][i])
    print(len(board_part), len(extra), i)
    board_parts.append(board_part)
    extra_parts.append(extra)
length = len(board_parts)
for i in range(19):
    #pk.dump(converted_fens[i * (length// 20):(i+1) * (length // 20)], open("converted_fens_773_split_"+str(i)+".pickle", "wb"), protocol  = pk.HIGHEST_PROTOCOL)
    pk.dump(board_parts[i * (length // 20):(i + 1) * (length // 20)],
            open("board_parts_" + str(i) + ".pickle", "wb"), protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(extra_parts[i * (length // 20):(i + 1) * (length // 20)],
            open("extra_parts_" + str(i) + ".pickle", "wb"), protocol=pk.HIGHEST_PROTOCOL)
pk.dump(board_parts[19 * (length // 20):(i + 1) * (length // 20)],
        open("board_parts_" + str(i) + ".pickle", "wb"), protocol=pk.HIGHEST_PROTOCOL)
pk.dump(extra_parts[19 * (length // 20):(i + 1) * (length // 20)],
        open("extra_parts_" + str(i) + ".pickle", "wb"), protocol=pk.HIGHEST_PROTOCOL)

# input1 = Input(shape = (12, 8, 8))
# input2 = Input(shape = (5,))
#
# initializer = GlorotUniform()
# regularizerl2 = L2(l2 = 0.1)
#
# conv_model = get_conv_model(regularizerl2)
# fc_model = fully_connected_part(initializer, regularizerl2)
# cnn_output = conv_model(input1)
# output = fc_model(Concatenate()([cnn_output, input2]))
# model = tf.keras.models.Model([input1, input2], output)
#
# adamOpti = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
#     name='Adam')
# model.compile(optimizer = adamOpti, loss = 'mse', metrics = ['mae', 'mse'])
#
# training_num = len(board_parts) * 4 // 5
# val_num = len(board_parts) - training_num
# # testing_num = len(board_parts) - training_num - val_num
# # print(training_num, val_num, testing_num)
#
# print(converted_fens[23])
# testing = [(converted_fens[i])[0] for i in range(training_num)]
# print(testing[0])
# X_train_board = np.array(board_parts[:training_num], dtype = np.float16)
# X_train_extra = np.array(extra_parts[:training_num], dtype = np.float16)
# X_val_board = np.array(board_parts[training_num:training_num+val_num], dtype = np.float16)
# X_val_extra = np.array(extra_parts[training_num:val_num], dtype = np.float16)
# Y_train = np.array(capped_evals[:training_num], dtype = np.float16)
# Y_val = np.array(capped_evals[training_num:training_num + val_num], dtype = np.float16)
#
# train_regression(model, [X_train_board, X_train_extra], Y_train, ([X_val_board, X_val_extra], Y_val))
#
# model.save_weights('2_part_network.h5')
# # plt.show()





