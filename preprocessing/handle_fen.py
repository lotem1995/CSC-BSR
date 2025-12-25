import numpy as np

piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 11, 'n': 12, 'b': 13, 'r': 14, 'q': 15, 'k': 16
    }

def fen_to_board_int(fen):
    """
    Parses a FEN string and returns an 8x8 numpy array of integers.

    Mapping:
    0  : Empty (.)
    1  : White Pawn   (P)       : Black Pawn   (p)
    2  : White Knight (N)       : Black Knight (n)
    3  : White Bishop (B)       : Black Bishop (b)
    4  : White Rook   (R)       : Black Rook   (r)
    5  : White Queen  (Q)       : Black Queen  (q)
    6  : White King   (K)       : Black King   (k)
    """


    # 1. Extract piece placement
    board_str = fen.split(' ')[0]
    ranks = board_str.split('/')

    board_rows = []

    for rank in ranks:
        row_data = []
        for char in rank:
            if char.isdigit():
                # Fill empty squares with 0
                num_empty = int(char)
                row_data.extend([0] * num_empty)
            else:
                # Map piece letter to integer
                row_data.append(piece_map[char])
        board_rows.append(row_data)

    return np.array(board_rows, dtype=np.int8)

