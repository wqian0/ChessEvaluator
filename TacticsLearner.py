import codecs

import pandas as pd
import numpy as np
import scipy as sp
from scipy import special
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from sklearn.utils import shuffle
import matplotlib.pylab as plt
import pickle as pk
import chess
import chess.pgn
import chess.engine

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
def read_all_games(tactics_file):
    FENs = []
    correct_moves = []
    curr = chess.pgn.read_game(tactics_file)
    count = 0
    while curr is not None:
        count += 1
        print(count)
        board = curr.board()
        moves = list(curr.mainline_moves())
        for i in range(len(moves) // 2):
            board.push(moves[i * 2])
            FENs.append(board.fen())
            board.push(moves[i * 2 + 1])
            correct_moves.append(moves[i * 2 + 1])
        curr = chess.pgn.read_game(tactics_file)
    return FENs, correct_moves


def convert_string_coordinate(input):
    file = input[0]
    rank = int(input[1]) - 1
    col = ord(file) - 97
    return (rank, col)
def convert_fen(fen, include_enpassant = False):
    parts = fen.split(" ")
    board = convert_board_part(parts[0])
    flattened_board = np.ndarray.flatten(board)
    if len(parts) != 6:
        print("invalid")
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
FENs, correct_moves = read_all_games(tactics_f)