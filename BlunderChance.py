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

pgns = open("lichess_db_standard_rated_2018-04.pgn")
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

def convert_fen_combined(fen, WhiteElo, BlackElo, min_elo, max_elo):
    parts = fen.split(" ")
    board_out = convert_board_part(parts[0])
    board_out[12] += 1 #constant ones for distinguishing between padding
    if parts[1] == 'w':
        board_out[13] += 1
    if "K" in parts[2]:
        board_out[14] += 1
    if "Q" in parts[2]:
        board_out[15] += 1
    if "k" in parts[2]:
        board_out[16] += 1
    if "q" in parts[2]:
        board_out[17] += 1
    normWhiteElo = (WhiteElo - min_elo) / (max_elo - min_elo)
    normBlackElo = (BlackElo - min_elo) / (max_elo - min_elo)
    board_out[18] += 1
    board_out[19] += 1
    board_out[18] *= normWhiteElo
    board_out[19] *= normBlackElo
    return board_out

def convert_board_and_elo_combined(board, WhiteElo, BlackElo, min_elo, max_elo):
    board_out = np.zeros((20, 8, 8))
    board_out[12] += 1 #constant ones for distinguishing between padding
    if board.turn is chess.WHITE:
        board_out[13] += 1
    if board.has_kingside_castling_rights(chess.WHITE):
        board_out[14] += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        board_out[15] += 1
    if board.has_kingside_castling_rights(chess.BLACK):
        board_out[16] += 1
    if board.has_queenside_castling_rights(chess.BLACK):
        board_out[17] += 1
    normWhiteElo = (WhiteElo - min_elo) / (max_elo - min_elo)
    normBlackElo = (BlackElo - min_elo) / (max_elo - min_elo)
    board_out[18] += 1
    board_out[19] += 1
    board_out[18] *= normWhiteElo
    board_out[19] *= normBlackElo
    for row in range(8):
        for col in range(8):
            squareIndex = row * 8 + col
            square=chess.SQUARES[squareIndex]
            piece = board.piece_at(square)
            if piece is not None:
                piece = piece.symbol()
                index = indices[piece]
                board_out[index][7 - row][col] = 1
    return board_out

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
    output = np.zeros((20, 8, 8), dtype = np.float16)
    modified_input = input
    for i in range(9):
        modified_input = modified_input.replace(str(i), '*'*i)
    modified_input = modified_input.replace("/", "")
    for i in range(8):
        for j in range(8):
            currLoc = 8*i + j #white starts from row 7, black from row 0
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

    X = Dense(256, kernel_regularizer=regularizerl2)(X)
    X = Activation("relu")(X)

    output = Dense(3, kernel_regularizer= regularizerl2, activation = 'softmax')(X)
    return tf.keras.Model(inputs = [X_in, X_extra], outputs = output)

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
    def __init__(self, dir, num_inputs, batch_size, elo_norm, min_elo, max_elo, shuffle=True):
        'Initialization'
        self.dir = dir
        self.elo_norm = elo_norm
        self.num_inputs = num_inputs
        self.batch_size = batch_size
        self.indices = np.arange(num_inputs)
        self.shuffle = shuffle
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.on_epoch_end()

    def __len__(self):
        return self.num_inputs

    def __getitem__(self, index):
        X, y = self.__data_generation(self.indices[index])
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def convert_board_part(input):  # One hot encoding for first 12 channels
        output = np.zeros((20, 8, 8), dtype=np.float16)
        modified_input = input
        for i in range(9):
            modified_input = modified_input.replace(str(i), '*' * i)
        modified_input = modified_input.replace("/", "")
        for i in range(8):
            for j in range(8):
                currLoc = 8 * i + j
                if modified_input[currLoc].isalpha():
                    output[indices[modified_input[currLoc]]][i][j] = 1
        return output

    def convert_fen_combined(self, fen, WhiteElo, BlackElo):
        parts = fen.split(" ")
        board_out = convert_board_part(parts[0])
        board_out[12] += 1  # constant ones for distinguishing between padding
        if parts[1] == 'w':
            board_out[13] += 1
        if "K" in parts[2]:
            board_out[14] += 1
        if "Q" in parts[2]:
            board_out[15] += 1
        if "k" in parts[2]:
            board_out[16] += 1
        if "q" in parts[2]:
            board_out[17] += 1
        normWhiteElo = (WhiteElo - self.min_elo) / (self.max_elo - self.min_elo)
        normBlackElo = (BlackElo - self.min_elo) / (self.max_elo - self.min_elo)
        board_out[18] += 1
        board_out[19] += 1
        board_out[18] *= normWhiteElo
        board_out[19] *= normBlackElo
        return board_out

    def __data_generation(self, index):
        fens = np.load(self.dir+'fens_'+str(index)+'.npy')
        outcomes = np.load(self.dir+'outcome_'+str(index)+'.npy')
        elos = np.load(self.dir + 'elos_' + str(index) + '.npy')
        boards = np.array([convert_fen_combined(fens[i], elos[i][0], elos[i][1], self.min_elo, self.max_elo) for i in range(len(fens))])
        return boards, outcomes

def read_all_games(pgnList, positions_cap, decisive_pad):
    FENs = []
    elos = []
    results = [] # 0, white wins, 1 draw, 2 black wins
    curr = chess.pgn.read_game(pgnList)
    decisives = 0
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
        whiteElo = curr.headers['WhiteElo']
        blackElo = curr.headers['BlackElo']
        # filters out time-controls below 3-minute blitz with 2 s inc, and games with less than 2 moves
        if len(moves) > 4 and plusIndex != -1 and int(curr.headers['TimeControl'][0:plusIndex]) >= 180 and int(curr.headers['TimeControl'][plusIndex + 1 :]) >= 2:
            print(gameNo, len(elos), result, whiteElo, blackElo, curr.headers['TimeControl'], decisives)
            for i in range(len(moves)):
                if np.random.rand() < ((i + 1)/len(moves)) ** 2:
                    FENs.append(board.fen())
                    elos.append([whiteElo, blackElo])
                    results.append(result)
                board.push(moves[i])
                if board.is_checkmate() or board.is_insufficient_material() or board.is_stalemate():
                    decisives += 1
                    for i in range(decisive_pad):
                        flipRand = np.random.rand()
                        if flipRand < .5:
                            FENs.append(board.fen())
                            results.append(result)
                        else:
                            FENs.append(board.mirror().fen())
                            results.append(2 - result)
                        elos.append([np.random.randint(1000, 3000), np.random.randint(1000, 3000)])
            FENs.append(board.fen())
            elos.append([whiteElo, blackElo])
            results.append(result)
        curr = chess.pgn.read_game(pgnList)
        gameNo += 1
    return FENs, elos, results


def get_best_move(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth, isWhite):
    bestScore = -100 if isWhite else 100
    bestMove = None
    print(board.legal_moves)
    for move in board.legal_moves:
        board.push(move)
        currScore = lookahead(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth - 1, -100, 100, not isWhite)
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

def eval(model, board, WhiteElo, BlackElo, min_elo, max_elo):
    input = convert_board_and_elo_combined(board, WhiteElo, BlackElo, min_elo, max_elo)
    prediction = np.array(model(np.array([input]))[0])
    return prediction[0]

def lookahead(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth, alpha, beta, isWhite):
    currHash = hash(board, isWhite, depth)
    if currHash in board_hashes:
        return board_hashes[currHash]
    if depth == 0 or not board.legal_moves:
        score = eval(model, board, WhiteElo, BlackElo, min_elo, max_elo)
        board_hashes[currHash] = score
        return score
    bestScore  = -99 if isWhite else 99

    for move in board.legal_moves:
        board.push(move)
        currScore = lookahead(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth - 1, alpha, beta, not isWhite)
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

    fens, elos, results = read_all_games(pgns, 40000000, 10)
    fens = np.array(fens)
    elos = np.array(elos, dtype = int)
    results = np.array(results, dtype = int)
    rand_indices = np.arange(len(fens))
    np.random.shuffle(rand_indices)
    fens = fens[rand_indices]
    elos = elos[rand_indices]
    results = results[rand_indices]

    training_num = len(fens) * 9 // 10
    val_num = len(fens) - training_num

    fens_train = fens[:training_num]
    fens_val = fens[training_num:]
    elos_train = elos[:training_num]
    elos_val = elos[training_num:]
    results_train = results[:training_num]
    results_val = results[training_num:]

    boards = []
    outcomes = []
    for i in range(training_num // 1024):
        print(i)
        np.save(open(train_dir + 'outcome_' + str(36565 + i) + ".npy", "wb"), np.array(results_train[i* 1024: (i + 1) * 1024]))
        np.save(open(train_dir + 'elos_' + str(36565 + i) + ".npy", "wb"), np.array(elos_train[i * 1024: (i + 1) * 1024]))
        np.save(open(train_dir + 'fens_'+str(36565 + i)+".npy", "wb"), fens_train[i* 1024: (i + 1) * 1024])

    for i in range(val_num // 1024):
        print(i)
        np.save(open(val_dir + 'outcome_' + str(4062 + i) + ".npy", "wb"), np.array(results_val[i* 1024: (i + 1) * 1024]))
        np.save(open(val_dir + 'elos_' + str(4062 + i) + ".npy", "wb"), np.array(elos_val[i * 1024: (i + 1) * 1024]))
        np.save(open(val_dir + 'fens_'+str(4062 + i)+".npy", "wb"), fens_val[i* 1024: (i + 1) * 1024])

    batchsize= 1024
    training_gen = DataGenerator(train_dir, 71721, batchsize, 3000, 1000, 3000)
    val_gen = DataGenerator(val_dir, 7968, batchsize, 3000, 1000, 3000)

    regularizerl2 = L2(l2 = 1e-6)
    #model = get_full_model_allconv(regularizerl2, 96, 8)
    #model = get_combined_model()
    model = tf.keras.models.load_model('fat_input.h5')
    # model.summary()
    # regAdam = tf.optimizers.Adam(learning_rate=.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False,
    #     name='Adam')
    # stoch = tf.keras.optimizers.SGD(learning_rate= .00004, momentum = .9, nesterov= True)
    # model.compile(optimizer = regAdam, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    #
    # history = model.fit(training_gen, validation_data = val_gen,
    #                     validation_batch_size=batchsize, validation_steps= 7968,  epochs = 1, batch_size= batchsize, verbose=1, steps_per_epoch= 71721, use_multiprocessing= True, workers = 8)
    # model.save('fat_input_3.h5')

    fen = 'r1bqkb1r/pppppppp/2n2n2/3P4/2P5/8/PP2PPPP/RNBQKBNR b KQkq - 0 3'
    board = chess.Board(fen)
    input = convert_board_and_elo_combined(board, 2800, 2800, 1000, 3000)

    print(model.predict(np.array([input])))

    print(get_best_move(model, board, 2100, 2100, 1000, 3000, 2, False))




