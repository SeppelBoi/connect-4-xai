# Pyton file with all connect 4 game functions and miniMax algorithm
# Using numba's @njit (=@jit(nopython=True)) to rewrite the functions to machine code
# Boosting the processing time for multiple uses of the functions
#imports
import numpy as np
#import pandas as pd
#import random
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
from numba import njit, int32
#import time

@njit
def create_board(board_size=(6,7)):
    board = np.full(board_size, 0)
    return board
    
@njit
def list_legal_moves(board):
    #Create list of legal moves by checking the top most row
    legal_moves = []
    for i in range(len(board[0])):
        if board[0][i] == 0: 
            legal_moves.append(i)
    return legal_moves
    
@njit
def make_move(board, column, player):
    for i in range(len(board) -1, -1, -1):
        if board[i, column] == 0:
            board[i, column] = player
            break
    return board, (i, column)
    
@njit
def remove_move(board, column):
    # Removes the last made move in the column
    for i in range(len(board)):
        if board[i, column] != 0:
            board[i, column] = 0
            break
    return board, (i, column)
    
@njit
def spots_left(board):
    # Returns the number of empty cells on the board
    return np.count_nonzero(board == 0)

@njit
def get_sections(board):
    # Returns all possible sections of 4 positions that could result in a win
    
    # Assign height and width of board and break if the shape is smaller than 4 in height or width
    h, w = board.shape
    if h < 4 and w < 4:
        print('Shape is too small to connect 4 pieces anywhere: ', board.shape)
        return np.empty((0, 4), dtype=np.int64)
    
    # Using @jit to translate the code to machine language the arrays and lists need to be static
    # Computating the amount of sections beforehand is necessary before filling them in
    number_of_sections = ((w - 3) * h) + ((h - 3) * w) + ((h - 3) * (w - 3)) + ((h - 3) * (w - 3))
    
    sections = np.empty((number_of_sections, 4), dtype=np.int64)
    section = 0
    
    # Horizontal Sections
    for j in range(w - 3):
        for i in range(h):
            sections[section, :] = np.array([board[i][j], board[i][j + 1], board[i][j + 2], board[i][j + 3]], dtype=np.int64)
            section += 1
    
    # Vertical Sections
    for i in range(h - 3):
        for j in range(w):
            sections[section, :] = np.array([board[i][j], board[i + 1][j], board[i + 2][j], board[i + 3][j]], dtype=np.int64)
            section += 1
    
    # Negative-sloped Diagonal Sections
    for i in range(h - 3):
        for j in range(w - 3):
            sections[section, :] = np.array([board[i][j], board[i + 1][j + 1], board[i + 2][j + 2], board[i + 3][j + 3]], dtype=np.int64)
            section += 1
    
    # Positive-sloped Diagonal Sections
    for i in range(h - 3, h):
        for j in range(w - 3):
            sections[section, :] = np.array([board[i][j], board[i - 1][j + 1], board[i - 2][j + 2], board[i - 3][j + 3]], dtype=np.int64)
            section += 1
    
    return sections

@njit
def section_score(section, player):
    # Assigns a score to a section based on how likely player is to win/lose
    score = 0
    player_count = 0
    not_player_count = 0
    empty = 0
    
    for position in section:
        if position == player:
            player_count += 1
        elif position == 0 - player:
            not_player_count += 1
        else:
            empty += 1
    
    # Examined section result in different counts
    if not_player_count == 3 and empty == 1:
        score -= 100
    if not_player_count == 2 and empty == 2:
        score -= 10
    if not_player_count == 1 and empty == 3:
        score -= 1
    if player_count == 3 and empty == 1:
        score += 100
    if player_count == 2 and empty == 2:
        score += 10
    if player_count == 1 and empty == 3:
        score += 1
    
    return score
    
@njit
def evaluation(board, player):
    # Function to assign a score to a board
    score = 0
    sections = get_sections(board)
    
    for section in sections:
        score += section_score(section, player)
        
    return score

@njit
def check_winner(board):
    # Checks if a player has won the game and returns the player number or 0
    sections = get_sections(board)
    for section in sections:
        if np.all(section == 1):
            return 1
        if np.all(section == -1):
            return -1
    return 0

# MiniMax algorithm, given a board, a depth, and a player to look at
# It returns a tuple consisting of the highest scoring move column and its score
@njit
def miniMax(board, depth=3, player=-1, isMax=False, alpha=-np.inf, beta=np.inf):
    # The recursion anchor. 
    # When a subtree has a winner, the board is full, or the depth reaches 0 a score will be returned
    winner = np.int32(check_winner(board))
    if winner == player:
        return np.int32(np.rint(1000 + (depth * 100) + (np.random.rand(1) * 100))[0]) , np.int32(-1)
    if winner == 0 - player:
        return np.int32(np.rint(-1000 - (depth * 100) - (np.random.rand(1) * 100))[0]), np.int32(-1)
    if spots_left(board) == 0:
        return 0, np.int32(-1)
    if depth == 0:
        return np.float64(evaluation(board, player)), np.int32(-1)

    # isMax == True: means the player you want to look at is the next player to make a move
    if isMax:
        bestMove, bestMoveColumn = -np.inf, np.random.choice(np.array(list_legal_moves(board)))
        for column in list_legal_moves(board):
            board_copy = np.copy(board)
            make_move(board_copy, column, player)
            val, _ = miniMax(board_copy, depth - 1, player, False, alpha, beta)
            
            # If the new move results in the best score take it
            if np.argmax(np.array([bestMove, val])) == 1:
                bestMove = val
                bestMoveColumn = column
                
#             remove_move(board_copy, column)
            
            alpha = max(alpha, val)
            if beta <= alpha:
                break
    
    # isMax == False: means the player you want to look at just made a move
    else:
        bestMove, bestMoveColumn = np.inf, np.random.choice(np.array(list_legal_moves(board)))
        for column in list_legal_moves(board):
            board_copy = np.copy(board)
            make_move(board_copy, column, 0 - player)
            val, _ = miniMax(board_copy, depth - 1, player, True, alpha, beta)
            
            # If the new move results in the worst score take it
            if np.argmin(np.array([bestMove, val])) == 1:
                bestMove = val
                bestMoveColumn = column
                
#             remove_move(board_copy, column)
            
            beta = min(beta, val)
            if beta <= alpha:
                break
                
    return np.float64(bestMove), np.int32(bestMoveColumn)