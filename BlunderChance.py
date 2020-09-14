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
import chess.polyglot
import chess.pgn
import chess.engine

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
#
# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

pgns = open("combined_lichess_elite.pgn")
# training_data = pd.read_csv("training_data_inc.csv")
# val_data = pd.read_csv("val_data_inc.csv")

train_dir = "C:/Users/billy/PycharmProjects/ChessEvaluator/training_data/"
val_dir = "C:/Users/billy/PycharmProjects/ChessEvaluator/val_data/"

# train_filename = "training_data_inc.csv"
# val_filename = "val_data_inc.csv"

fen_all_pieces = 'KQRBNPkqrbnp'
indices = {}
board_hashes = {}
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

def convert_board_and_elo(board, WhiteElo, BlackElo, elo_norm):
    board_out = np.zeros((6, 8, 8))
    extra_out = np.zeros(7)
    if board.turn is chess.WHITE:
        extra_out[0] = 1
    else:
        extra_out[0] = -1
    if board.has_kingside_castling_rights(chess.WHITE):
        extra_out[1] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        extra_out[2] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        extra_out[3] = -1
    if board.has_queenside_castling_rights(chess.BLACK):
        extra_out[4] = -1
    extra_out[5] = WhiteElo / elo_norm
    extra_out[6] = BlackElo / elo_norm
    for row in range(8):
        for col in range(8):
            squareIndex = row * 8 + col
            square=chess.SQUARES[squareIndex]
            piece = board.piece_at(square)
            if piece is not None:
                piece = piece.symbol()
                index = indices[piece] % 6
                if piece.isupper():
                    board_out[index][row][col] = 1
                else:
                    board_out[index][row][col] = -1
    return board_out, extra_out

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

def get_full_model(regularizerl2, filters, residuals):
    X_in = Input(shape=(6, 8, 8))
    X_extra = Input(shape=(7,))
    X = Conv2D(filters, (3, 3), activation='relu', input_shape=(6, 8, 8), padding='same',
               data_format="channels_first", kernel_regularizer=regularizerl2)(X_in)
    X = BatchNormalization(axis = 1)(X)
    X = Activation("relu")(X)
    for i in range(residuals):
        X = create_residual_block(X, filters, regularizerl2)
    X = Conv2D(16, (1, 1), activation='relu', input_shape=(6, 8, 8), padding='same',
           data_format="channels_first", kernel_regularizer=regularizerl2)(X)
    X = BatchNormalization(axis = 1)(X)
    X = Activation("relu")(X)
    X = Flatten()(X)
    X = Concatenate()([X, X_extra])

    X = Dense(512, kernel_regularizer= regularizerl2)(X)
    X = Activation("relu")(X)

    X = Dense(256, kernel_regularizer=regularizerl2)(X)
    X = Activation("relu")(X)

    output = Dense(3, kernel_regularizer= regularizerl2, activation = 'softmax')(X)
    return tf.keras.Model(inputs = [X_in, X_extra], outputs = output)

def get_full_model_NHWC(regularizerl2, residuals):
    X_in = Input(shape=(8, 8, 6))
    X_extra = Input(shape=(7,))
    X = Conv2D(80, (3, 3), activation='relu', input_shape=(8, 8, 6), padding='same',
               data_format="channels_last", kernel_regularizer=regularizerl2)(X_in)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    for i in range(residuals):
        X = create_residual_block(X, regularizerl2)
    X = Conv2D(16, (1, 1), activation='relu', input_shape=(8, 8, 6), padding='same',
           data_format="channels_last", kernel_regularizer=regularizerl2)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Flatten()(X)
    X = Concatenate()([X, X_extra])
    X = Dense(256, kernel_regularizer= regularizerl2)(X)
    X = Activation("relu")(X)
    output = Dense(3, kernel_regularizer= regularizerl2, activation = 'softmax')(X)
    return tf.keras.Model(inputs = [X_in, X_extra], outputs = output)

def fully_connected_part(regularizerl2):
    model = Sequential()
    model.add(Dense(1024, activation='relu', kernel_regularizer = regularizerl2))
    # model.add(Dense(256, activation='relu', kernel_regularizer = regularizerl2))
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

def convert_board_part_NHWC(input): # -1, 0, 1 encoding
    output = np.zeros((8, 8, 6), dtype=np.float16)
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
                    output[i][j][index] = 1
                else:
                    output[i][j][index] = -1
    return output

def convert_fen3(fen, compressed = False, flip = False):
    parts = fen.split(" ")
    if compressed:
        board = convert_board_part_NHWC(parts[0])
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


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dir, num_inputs, batch_size, elo_norm, shuffle=True):
        'Initialization'
        self.dir = dir
        self.elo_norm = elo_norm
        self.num_inputs = num_inputs
        self.batch_size = batch_size
        self.indices = np.arange(num_inputs)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return self.num_inputs

    def __getitem__(self, index):
        X, y = self.__data_generation(self.indices[index])
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, index):
        boards = np.load(self.dir+'board_'+str(index)+'.npy')
        extras = np.array(np.load(self.dir+'extra_'+str(index)+'.npy'), dtype = np.float16)
        extras[:, 5] /= self.elo_norm
        extras[:, 6] /= self.elo_norm
        outcomes = np.load(self.dir+'outcome_'+str(index)+'.npy')
        return [boards, extras], outcomes

def read_all_games(pgnList, positions_cap):
    FENs = []
    elos = []
    results = [] # 0, white wins, 1 draw, 2 black wins
    curr = chess.pgn.read_game(pgnList)
    gameNo = 0
    while curr is not None and len(elos) < positions_cap:
        board = curr.board()
        moves = list(curr.mainline_moves())
        result = 0
        if curr.headers['Result'] == '1/2-1/2':
            result = 1
        if curr.headers['Result'] == '0-1':
            result = 2
        plusIndex = curr.headers['TimeControl'].find('+')
        # filters out time-controls below 3-minute blitz with 2 s inc, and games with less than 2 moves
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


def get_best_move(model, board, WhiteElo, BlackElo, elo_norm, depth, isWhite):
    bestScore = -100 if isWhite else 100
    bestMove = None
    for move in board.legal_moves:
        board.push(move)
        currScore = lookahead(model, board, WhiteElo, BlackElo, elo_norm, depth - 1, -100, 100, not isWhite)
        board_hashes[hash(board, not isWhite, depth)] = currScore
        if isWhite and currScore > bestScore:
            bestScore = currScore
            bestMove = move
        elif not isWhite and currScore < bestScore:
            bestScore = currScore
            bestMove = move
        board.pop()
        print(bestScore, bestMove, currScore, move, len(board_hashes))
    return bestMove

def hash(board, isWhite, depth):
    return str(board) +str(isWhite) +str(depth)

def eval(model, board, WhiteElo, BlackElo, elo_norm):
    board_part, extra = convert_board_and_elo(board, WhiteElo, BlackElo, elo_norm)
    prediction = np.array(model([np.array([board_part]), np.array([extra])])[0])
    return prediction[0]

def lookahead(model, board, WhiteElo, BlackElo, elo_norm, depth, alpha, beta, isWhite):
    currHash = hash(board, isWhite, depth)
    if currHash in board_hashes:
        return board_hashes[currHash]
    if depth == 0 or not board.legal_moves:
        score = eval(model, board, WhiteElo, BlackElo, elo_norm)
        board_hashes[currHash] = score
        return score
    bestScore  = -99 if isWhite else 99

    for move in board.legal_moves:
        board.push(move)
        currScore = lookahead(model, board, WhiteElo, BlackElo, elo_norm, depth - 1, alpha, beta, not isWhite)
        board_hashes[hash(board, isWhite, depth)] = currScore
        if isWhite:
            bestScore = max(bestScore, currScore)
            alpha = max(bestScore, alpha)
        else:
            bestScore = min(bestScore, currScore)
            beta = min(bestScore, beta)
        board.pop()
        if beta <= alpha:
            return bestScore
    return bestScore

if __name__ == "__main__":

    # fens, elos, results = read_all_games(pgns, 30000000)
    # fens = np.array(fens)
    # elos = np.array(elos)
    # results = np.array(results)
    # rand_indices = np.arange(len(fens))
    # np.random.shuffle(rand_indices)
    # fens = fens[rand_indices]
    # elos = elos[rand_indices]
    # results = results[rand_indices]
    #
    # training_num = len(fens) * 9 // 10
    # val_num = len(fens) - training_num
    #
    # fens_train = fens[:training_num]
    # fens_val = fens[training_num:]
    # elos_train = elos[:training_num]
    # elos_val = elos[training_num:]
    # results_train = results[:training_num]
    # results_val = results[training_num:]
    #
    # board_parts = []
    # extra_parts = []
    # outcomes = []
    # for i in range(training_num // 1024):
    #     print(i)
    #     for j in range(1024):
    #         board, extra = convert_fen2(fens_train[i * 1024 + j], compressed = True)
    #         extra = np.concatenate([extra, np.array(elos_train[i * 1024 + j])])
    #         board_parts.append(board)
    #         extra_parts.append(extra)
    #         outcomes.append(results_train[i * 1024 + j])
    #     np.save(open(train_dir + 'board_'+str(89015 + i)+".npy", "wb"), np.array(board_parts))
    #     np.save(open(train_dir + 'extra_' + str(89015 + i) + ".npy", "wb"), np.array(extra_parts))
    #     np.save(open(train_dir + 'outcome_' + str(89015 + i) + ".npy", "wb"), np.array(outcomes))
    #     board_parts = []
    #     extra_parts = []
    #     outcomes = []
    #
    # for i in range(val_num // 1024):
    #     print(i)
    #     for j in range(1024):
    #         board, extra = convert_fen2(fens_val[i * 1024 + j], compressed= True)
    #         extra = np.concatenate([extra, np.array(elos_val[i * 1024 + j])])
    #         board_parts.append(board)
    #         extra_parts.append(extra)
    #         outcomes.append(results_val[i * 1024 + j])
    #     np.save(open(val_dir + 'board_'+str(10703 + i)+".npy", "wb"), np.array(board_parts))
    #     np.save(open(val_dir + 'extra_' + str(10703 + i) + ".npy", "wb"), np.array(extra_parts))
    #     np.save(open(val_dir + 'outcome_' + str(10703 + i) + ".npy", "wb"), np.array(outcomes))
    #     board_parts = []
    #     extra_parts = []
    #     outcomes = []

    batchsize= 1024
    training_gen = DataGenerator(train_dir, 106261, batchsize, 3000)
    val_gen = DataGenerator(val_dir, 12619, batchsize, 3000)

    regularizerl2 = L2(l2 = 0)
    model = get_full_model(regularizerl2, 96, 8)
    #model = get_combined_model()
    model.summary()
    decayedAdam = AdamW(weight_decay = 1e-6, learning_rate = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='AdamW')
    regAdam = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False,
        name='Adam')
    stoch = tf.keras.optimizers.SGD(learning_rate= .00004, momentum = .9, nesterov= True)
    model.compile(optimizer = regAdam, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    # model.load_weights('fat model_with_big_epoch_2.h5')
    history = model.fit(training_gen, validation_data = val_gen,
                        validation_batch_size=batchsize, validation_steps= 12619 // 25,  epochs = 25, batch_size= batchsize, verbose=1, steps_per_epoch= 106261 // 25, use_multiprocessing= True, workers = 8)
    model.save_weights('new_model.h5')
    # plt.plot(history.history['sparse_categorical_accuracy'])
    # plt.plot(history.history['val_sparse_categorical_accuracy'])
    # plt.show()

    # board = chess.Board()
    # board.set_fen('3r3k/pp1b3Q/2pp4/8/2P1BP2/1P4P1/P5K1/8 b - - 2 4')
    # board_part, extra = convert_board_and_elo(board, 2100, 2100, 3000)
    # print(model.predict([np.array([board_part]), np.array([extra])]))

    # print(get_best_move(model, board, 2100, 2100, 3000, 3, True))




