# ChessEvaluator

A neural network that predicts the most likely outcome of a chess game (white wins, draw, black wins) from a given starting position, based on the two players' chess ratings. This network was trained on over 50 million positions drawn from human games found in the [lichess game database](https://database.lichess.org/). 

Positions that can be objectively evaluated as winning for one side by a traditional chess engine (i.e., Stockfish) are sometimes predicted to be more favorable for the opposing side. This is sometimes a result of the rating difference between the two players, but can also happen due to one side having a more "difficult" position to play. 
