"""
Tic Tac Toe Player
"""

import math
import copy
import random

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # X makes first move
    if board == initial_state():
        return X
    if terminal(board):
        return None
    # loop over every board space, if even number of moves have been made it's X turn
    moves = 0
    for row in board:
        for cell in row:
            if cell:
                moves += 1
    if moves % 2 == 0:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # initialize set of possible actions
    possible_moves = set()
    # loop over every board space and add empty locations to possible moves
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == EMPTY:
                possible_moves.add((i, j))
    return possible_moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise ValueError("Invalid move")
    # unpack action tuple (i, j)
    i, j = action
    # only modify a copy of the original board as algorithm needs to consider many board states
    result_board = copy.deepcopy(board)
    result_board[i][j] = player(board)
    return result_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check that a row is filled with all X or all O
    for row in board:
        if row[1:] == row[:-1] and row[1] != EMPTY:
            return row[1]
    # representation of all columns and diagonals on board that could result in a win state
    columns_and_diagonals = [[board[0][0], board[1][0], board[2][0]],
                             [board[0][1], board[1][1], board[2][1]],
                             [board[0][2], board[1][2], board[2][2]],
                             [board[0][0], board[1][1], board[2][2]],
                             [board[0][2], board[1][1], board[2][0]]]
    # check that a column or diagonal is filled with all X or all O
    for line in columns_and_diagonals:
        if line[1:] == line[:-1] and line[1] != EMPTY:
            return line[1]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # call winner function to see if game is won
    if winner(board):
        return True
    # check if any cells are still empty
    for row in board:
        if EMPTY in row:
            return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns one optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    # X is the max player
    if player(board) == X:
        # intialize list of optimal moves
        moves = []
        best_value = max_value(board)
        # loop over possible actions
        for action in actions(board):
            # find possible actions for which min value of the result matches max value for current state
            if min_value(result(board, action)) == best_value:
                # add action to moves list
                moves.append(action)
        # randomly choose a move from moves list
        return random.choice(moves)
    # O player is inverse of X
    if player(board) == O:
        moves = []
        best_value = min_value(board)
        for action in actions(board):
            if max_value(result(board, action)) == best_value:
                moves.append(action)
        return random.choice(moves)


def max_value(board, alpha=-math.inf, beta=math.inf):
    """
    Returns the maximum utility of a given board state
    """
    if terminal(board):
        return utility(board)
    value = -math.inf
    for action in actions(board):
        value = max(value, min_value(result(board, action), alpha, beta))
        if value >= beta:
            return value
        alpha = max(value, alpha)
    return value


def min_value(board, alpha=-math.inf, beta=math.inf):
    """
    Returns the minimum utility of a given board state
    """
    if terminal(board):
        return utility(board)
    value = math.inf
    for action in actions(board):
        value = min(value, max_value(result(board, action), alpha, beta))
        if value <= alpha:
            return value
        beta = min(value, beta)
    return value
