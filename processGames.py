import chess.pgn
import numpy as np
import tensorflow as tf

fen_all_pieces = 'KQRBNPkqrbnp'
indices = {}
for i in range(len(fen_all_pieces)):
    indices[fen_all_pieces[i]] = i

def read_all_games(pgnList, positions_cap, base_prob):
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
        whiteElo = curr.headers['WhiteElo']
        blackElo = curr.headers['BlackElo']
        # filters out time-controls below 3-minute blitz with 2 s inc, and games with less than 2 moves
        if len(moves) > 4 and plusIndex != -1 and int(curr.headers['TimeControl'][0:plusIndex]) >= 180 and \
                int(curr.headers['TimeControl'][plusIndex + 1 :]) >= 2:
            print(gameNo, len(elos), result, whiteElo, blackElo, curr.headers['TimeControl'])
            currCount = 0
            for i in range(len(moves)):
                if np.random.rand() < base_prob + (1 - base_prob) * ((i + 1)/len(moves)) ** 2:
                    currCount += 1
                    FENs.append(board.fen())
                board.push(moves[i])
            FENs.append(board.fen())
            elos.extend([[whiteElo, blackElo] for i in range(currCount + 1)])
            results.extend([result for i in range(currCount + 1)])
        curr = chess.pgn.read_game(pgnList)
        gameNo += 1
    return FENs, elos, results

def writePositionsToFile(pgn_list, positions_cap, base_prob, train_frac, batch_size,
                         train_dir, val_dir, train_start, val_start):
    fens, elos, results = read_all_games(pgn_list, positions_cap, base_prob)
    fens = np.array(fens)
    elos = np.array(elos, dtype = int)
    results = np.array(results, dtype = int)
    rand_indices = np.arange(len(fens))
    np.random.shuffle(rand_indices)
    fens = fens[rand_indices]
    elos = elos[rand_indices]
    results = results[rand_indices]

    training_num = int(len(fens) * train_frac)
    val_num = len(fens) - training_num

    fens_train = fens[:training_num]
    fens_val = fens[training_num:]
    elos_train = elos[:training_num]
    elos_val = elos[training_num:]
    results_train = results[:training_num]
    results_val = results[training_num:]

    for i in range(training_num // batch_size):
        print(i)
        np.save(open(train_dir + 'outcome_' + str(train_start + i) + ".npy", "wb"),
                np.array(results_train[i* batch_size: (i + 1) * batch_size]))
        np.save(open(train_dir + 'elos_' + str(train_start + i) + ".npy", "wb"),
                np.array(elos_train[i * batch_size: (i + 1) * batch_size]))
        np.save(open(train_dir + 'fens_'+str(train_start + i)+".npy", "wb"),
                fens_train[i* batch_size: (i + 1) * batch_size])

    for i in range(val_num // batch_size):
        print(i)
        np.save(open(val_dir + 'outcome_' + str(val_start + i) + ".npy", "wb"),
                np.array(results_val[i* batch_size: (i + 1) * batch_size]))
        np.save(open(val_dir + 'elos_' + str(val_start + i) + ".npy", "wb"),
                np.array(elos_val[i * batch_size: (i + 1) * batch_size]))
        np.save(open(val_dir + 'fens_'+str(val_start + i)+".npy", "wb"),
                fens_val[i* batch_size: (i + 1) * batch_size])

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

def convert_fen_combined(fen, WhiteElo, BlackElo, min_elo, max_elo):
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
    normWhiteElo = (WhiteElo - min_elo) / (max_elo - min_elo)
    normBlackElo = (BlackElo - min_elo) / (max_elo - min_elo)
    board_out[18] += 1
    board_out[19] += 1
    board_out[18] *= normWhiteElo
    board_out[19] *= normBlackElo
    return board_out

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

    def __data_generation(self, index):
        fens = np.load(self.dir+'fens_'+str(index)+'.npy')
        outcomes = np.load(self.dir+'outcome_'+str(index)+'.npy')
        elos = np.load(self.dir + 'elos_' + str(index) + '.npy')
        boards = np.array([convert_fen_combined(fens[i], elos[i][0], elos[i][1], self.min_elo, self.max_elo) for i in range(len(fens))])
        return boards, outcomes

if __name__ == "__main__":
    train_dir = "C:/Users/billy/PycharmProjects/ChessEvaluator/training_data/"
    val_dir = "C:/Users/billy/PycharmProjects/ChessEvaluator/val_data/"
    pgns = open("lichess_db_standard_rated_2018-05.pgn")
    writePositionsToFile(pgns, 40000000, .1, .9, 1024,  train_dir, val_dir, 92284, 10253)

