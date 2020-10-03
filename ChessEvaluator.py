
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import pickle as pk
import chess
import chess.polyglot
import chess.pgn
import chess.engine
import processGames as pg
from chess.engine import Cp
import asyncio
import time

board_hashes= {}

class BoardNode:
    def __init__(self, board):
        self.isWhite = board.turn == chess.WHITE
        self.depth = 0
        self.lower = -1
        self.upper = 1
        self.move = None
        self.hash = hasher(board)
    def __hash__(self):
        return self.hash

def hasher(board):
    isWhite = board.turn == chess.WHITE
    return chess.polyglot.zobrist_hash(board) * 10 + isWhite

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
                index = pg.indices[piece]
                board_out[index][7 - row][col] = 1
    return board_out

def IterDeep(model, board, WhiteElo, BlackElo, min_elo, max_elo, max_depth, max_time):
    start_time = time.time()
    curr_time = time.time()
    d = 1
    move = None
    while d <= max_depth and curr_time - start_time < max_time:
        print("DEPTH", d)
        move = get_best_move(model, board, WhiteElo, BlackElo, min_elo, max_elo, d)
        d += 1
        curr_time = time.time()
    return move

def predict(board, WhiteElo, BlackElo, min_elo, max_elo):
    input = convert_board_and_elo_combined(board, WhiteElo, BlackElo, min_elo, max_elo)
    return model.predict(np.array([input]))


def eval(model, board, WhiteElo, BlackElo, min_elo, max_elo):
    input = convert_board_and_elo_combined(board, WhiteElo, BlackElo, min_elo, max_elo)
    prediction = np.array(model(np.array([input]))[0])
    return prediction[0] - prediction[2]

def get_best_move(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth):
    isWhite = (board.turn == chess.WHITE)
    bestScore = -1 if isWhite else 1
    bestMove = None
    for move in board.legal_moves:
        board.push(move)
        currScore = lookahead(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth - 1, -1, 1)
        if isWhite and currScore > bestScore:
            bestScore = currScore
            bestMove = move
        elif not isWhite and currScore < bestScore:
            bestScore = currScore
            bestMove = move
        board.pop()
        print(bestScore, bestMove, currScore, move, len(board_hashes))
    return bestMove

def lookahead(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth, alpha, beta):
    currHash = hasher(board)
    if currHash in board_hashes:
        n = board_hashes[currHash]
        if n.depth >= depth:
            if n.lower >= beta:
                return n.lower
            if n.upper <= alpha:
                return n.upper
            alpha = max(alpha, n.lower)
            beta = min(beta, n.upper)
    if depth == 0 or not board.legal_moves:
        if board.is_game_over():
            if board.result() == '1-0':
                g = 1
            if board.result() == '1/2-1/2':
                g = 0
            if board.result() == '0-1':
                g = -1
        else:
            g = eval(model, board, WhiteElo, BlackElo, min_elo, max_elo)
    elif board.turn == chess.WHITE:
        a, g = alpha, -1
        for move in board.legal_moves:
            if g >= beta:
                break
            board.push(move)
            currScore = lookahead(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth - 1, a, beta)
            board.pop()
            if currScore > g:
                g = currScore
                bestMove = move
            a = max(a, g)
    else:
        b, g = beta, 1
        for move in board.legal_moves:
            if g <= alpha:
                break
            board.push(move)
            currScore = lookahead(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth - 1, alpha, b)
            board.pop()
            if currScore < g:
                g = currScore
                bestMove = move
            b = min(b, g)
    n = BoardNode(board)
    board_hashes[currHash] = n
    n.depth = depth
    if g <= alpha:
        n.upper = g
    if g > alpha and g < beta:
        n.upper = g
        n.lower = g
    if g >= beta:
        n.lower = g
    return g

if __name__ == "__main__":
    model = tf.keras.models.load_model('newest_bigger_model2.h5')
    board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    print(predict(board, 2200, 2200, 1000, 3000))
    #print(get_best_move(model, board, 2200, 2200, 1000, 3000, 4))
    timeStart = time.time()
    print(IterDeep(model, board, 2200, 2200, 1000, 3000, 10, 20))
    print(time.time() - timeStart)



