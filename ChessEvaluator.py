
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

board_hashes= {}

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

def get_best_move(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth, isWhite):
    bestScore = -100 if isWhite else 100
    bestMove = None
    print(board.legal_moves)
    for move in board.legal_moves:
        board.push(move)
        currScore = lookahead(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth - 1, -1, 1, not isWhite)
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

def predict(board, WhiteElo, BlackElo, min_elo, max_elo):
    input = convert_board_and_elo_combined(board, WhiteElo, BlackElo, min_elo, max_elo)
    return model.predict(np.array([input]))

def hash(board, isWhite, depth):
    return str(board) +str(isWhite) +str(depth)

def eval(model, board, WhiteElo, BlackElo, min_elo, max_elo):
    input = convert_board_and_elo_combined(board, WhiteElo, BlackElo, min_elo, max_elo)
    prediction = np.array(model(np.array([input]))[0])
    return prediction[0] - prediction[2]

def lookahead(model, board, WhiteElo, BlackElo, min_elo, max_elo, depth, alpha, beta, isWhite):
    currHash = hash(board, isWhite, depth)
    if currHash in board_hashes:
        return board_hashes[currHash]
    if depth == 0 or not board.legal_moves:
        if board.is_game_over():
            if board.result() == '1-0':
                score = 1
            if board.result() == '1/2-1/2':
                score = 0
            if board.result() == '0-1':
                score = -1
        else:
            score = eval(model, board, WhiteElo, BlackElo, min_elo, max_elo)
        board_hashes[currHash] = score
        return score
    bestScore  = -1 if isWhite else 1
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
    model = tf.keras.models.load_model('newest_bigger_model.h5')
    board = chess.Board('rnbqkb1r/pppppppp/5n2/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2')
    print(predict(board, 2100, 2100, 1000, 3000))
    print(get_best_move(model, board, 2200, 2200, 1000, 3000, 4, False))




