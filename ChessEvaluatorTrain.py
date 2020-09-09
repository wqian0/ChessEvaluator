import codecs
import gc
import pandas as pd
import numpy as np
import scipy as sp
from scipy import special
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Concatenate, Reshape, concatenate, LeakyReLU
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

file = open("chessData.csv", "r")
chess_data = pd.read_csv(file, encoding = 'utf_8_sig')
#
# tactics_f = open("tactics.pgn", "r")

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

def convert_board_part_compressed(input):
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
        board_input, extras = convert_fen2(temp_board.fen(), compressed= True)
        curr_score = model.predict([np.array([board_input]), np.array([extras])])[0][0]
        scores[index] = curr_score
        index += 1
    if board.turn == chess.WHITE:
        bestIndex = np.argmax(scores)
    else:
        bestIndex = np.argmin(scores)
    #print(moves, scores)
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
    def inner_model():
        model = Sequential()
        model.add(Dense(800, input_dim=389))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(400))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(200))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(1))
        return model

    input1 = Input(shape=(6, 8, 8))
    input2 = Input(shape=(5,))
    input = Concatenate()([Flatten()(input1), input2])
    seq_model = inner_model()
    model = tf.keras.models.Model(inputs = [input1, input2], outputs = seq_model(input))
    return model

def get_conv_model(regularizerl2):
    model = Sequential()
    model.add(
        Conv2D(16, (1, 1), activation='relu', input_shape=(6, 8, 8), padding='same', data_format="channels_first", kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(
        Conv2D(128, (7, 7), activation='relu', input_shape=(6, 8, 8), padding='same',data_format="channels_first", kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(
        Conv2D(128, (5, 5), activation='relu', input_shape=(6, 8, 8), padding='same',data_format="channels_first", kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(
        Conv2D(128, (3, 3), activation='relu', input_shape=(6, 8, 8), padding='same',data_format="channels_first", kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())

    return model

def fully_connected_part(initializer, regularizerl2):
    model = Sequential()
    model.add(Dense(1024, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizerl2))
    model.add(Dense(1))
    return model

def get_combined_model():
    input1 = Input(shape = (6,8,8))
    input2 = Input(shape = (5,))

    initializer = GlorotUniform()
    regularizerl2 = L2(l2 = 1e-6)

    conv_model = get_conv_model(regularizerl2)
    fc_model = fully_connected_part(initializer, regularizerl2)
    cnn_output = Flatten()(conv_model(input1))
    output = fc_model(Concatenate(axis = -1)([cnn_output, input2]))
    model = tf.keras.models.Model(inputs = [input1, input2], outputs = output)
    model.summary()
    return model


def generate_from_df(df, Y, batchsize, fliprate):
    while True:
        board_parts = []
        extra_parts = []
        indices = np.random.choice(len(Y), batchsize)
        evals = Y[indices]
        for i in range(len(indices)):
            flip = False
            if np.random.rand() < fliprate:
                flip = True
            board_part, extra_part = convert_fen2(df['FEN'][indices[i]], compressed= True, flip = flip)
            board_parts.append(board_part)
            extra_parts.append(extra_part)
            if flip:
                evals[i] *= -1
        yield [np.asarray(board_parts), np.asarray(extra_parts)], evals

def train_classification(model, train_x, train_y, val_data):
    return model.fit(train_x, train_y, epochs = 5, verbose = 1, validation_data = val_data, batch_size = 128)

def train_regression(model, train_x, train_y, val_data, epochs, batch_size, weights = None):
    return model.fit(train_x, train_y, epochs = epochs, steps_per_epoch= len(chess_data) // (2 * batch_size), verbose = 1, validation_data = val_data, sample_weight= weights, batch_size= batch_size)


if __name__ == '__main__':
    fen_all_pieces = 'KQRBNPkqrbnp'
    indices = {}
    for i in range(len(fen_all_pieces)):
        indices[fen_all_pieces[i]] = i
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
    #
    # board_parts = []
    # extra_parts = []
    #
    # for i in range(len(chess_data['FEN'])):
    #     board_part, extra = convert_fen2(chess_data['FEN'][i], compressed= True)
    #     print(len(board_part), len(extra), i, extra)
    #     board_parts.append(board_part)
    #     extra_parts.append(extra)
    # length = len(board_parts)
    # for i in range(18, 19):
    #     #pk.dump(converted_fens[i * (length// 20):(i+1) * (length // 20)], open("converted_fens_773_split_"+str(i)+".pickle", "wb"), protocol  = pk.HIGHEST_PROTOCOL)
    #     pk.dump(board_parts[i * (length // 20):(i + 1) * (length // 20)],
    #             open("board_parts_comp_" + str(i) + ".pickle", "wb"), protocol=pk.HIGHEST_PROTOCOL)
    #     pk.dump(extra_parts[i * (length // 20):(i + 1) * (length // 20)],
    #             open("extra_parts_comp_" + str(i) + ".pickle", "wb"), protocol=pk.HIGHEST_PROTOCOL)
    # pk.dump(board_parts[19 * (length // 20):],
    #         open("board_parts_comp_" + str(19) + ".pickle", "wb"), protocol=pk.HIGHEST_PROTOCOL)
    # pk.dump(extra_parts[19 * (length // 20):],
    #         open("extra_parts_comp_" + str(19) + ".pickle", "wb"), protocol=pk.HIGHEST_PROTOCOL)



    evals = []
    for i in range(16):
        evals.extend(pk.load(open("evals_"+str(i)+".pickle", "rb")))

    val_indices_keep = []
    val_boards = []
    val_extras = []
    val_evals = []
    for i in range(16, 18):
        val_boards.extend(pk.load(open("board_parts_comp_"+str(i)+".pickle", "rb")))
        val_extras.extend(pk.load(open("extra_parts_comp_" + str(i) + ".pickle", "rb")))
        val_evals.extend(pk.load(open("evals_"+str(i)+".pickle", "rb")))

    evals = np.clip(np.asarray(evals, dtype = np.float16), a_min= -50, a_max = 50)

    val_boards = np.asarray(val_boards)
    val_extras = np.asarray(val_extras)
    val_evals = np.clip(np.asarray(val_evals), a_min= -50, a_max = 50)

    val_boards = val_boards[val_indices_keep]
    val_extras = val_extras[val_indices_keep]
    val_evals = val_evals[val_indices_keep]


    mean = np.mean(capped_evals)
    min_val = np.min(capped_evals)
    max_val = np.max(capped_evals)
    evals = 2 * (evals - min_val) / (max_val - min_val) - 1
    val_evals = 2 * (val_evals - min_val) / (max_val - min_val) - 1

    print(val_evals)

    model = get_combined_model()
    decayedAdamOpti = AdamW(weight_decay = 1e-5, learning_rate = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='AdamW')
    adamOpti = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')
    stoch = tf.keras.optimizers.SGD(learning_rate= .001, momentum = .8, nesterov= True)
    model.compile(optimizer = adamOpti, loss = 'mse', metrics = ['mae', 'mse'])
    train_regression(model, generate_from_df(chess_data, evals, 1024, .5), None, ([val_boards, val_extras], val_evals), 5, 1024)

    model.save_weights('deepdeepnetwork.h5')
    prediction = (model.predict([val_boards, val_extras])  + 1)* (max_val - min_val) / 2 + min_val
    plt.scatter(prediction, (val_evals + 1) * (max_val - min_val) / 2 + min_val, s = 5, alpha = .02)
    #plt.hist2d(prediction[:,0], (val_evals + 1) * (max_val - min_val) / 2 + min_val, (500, 500), cmax = 40)
    plt.colorbar()
    plt.show()





