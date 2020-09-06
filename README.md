# ChessEvaluator

A neural network that predicts the most likely outcome of a chess game (white wins, draw, black wins) from a given starting position, based on the two players' chess ratings. This network was trained on over 50 million positions drawn from human games found in the [lichess game database](https://database.lichess.org/). 

Positions that can be objectively evaluated as winning for one side by a traditional chess engine (i.e., Stockfish) are sometimes predicted to be more favorable for the opposing side. This is sometimes a result of the rating difference between the two players, but can also happen due to one side having a more "difficult" position to play from a human perspective. 

Here is an example of this from an online game I played:

[White "elitechicken"]
[Black "GM Akshayraj_Kore"]
[Site "Chess.com"]
[WhiteElo "2139"]
[BlackElo "2477"]
<img src="https://github.com/wqian0/ChessEvaluator/blob/master/chesspos_1.png" width="400" height="400"/>

Stockfish at depth 38 evaluates this position to be slightly favorable for white (+0.64), likely because of the slight material advantage. However, black's pieces are much more active, and white's king is exposed, making the defense quite difficult even though the position is objectively holdable. I went on to lose this position quite quickly. 

Indeed, the ANN predicts that, even if the white and black players were equally matched in terms of ratings, this position still favors black. 

The predictions made by this model can also be used as metrics for position sharpness/complexity. Positions that are evaluated to have around equal winning chances for white and black, but have extremely low drawing chances, can be interpreted as being sharp. 
