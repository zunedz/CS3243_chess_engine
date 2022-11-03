from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import sys
from typing import Deque, Dict, List, Set, Tuple
from random import choice

### IMPORTANT: Remove any print() functions or rename any print functions/variables/string when submitting on CodePost
### The autograder will not run if it detects any print function.

ROWS, COLS = 7, 7 
MAX_DEPTH = 4  

#############################################################################
######## Piece Logic
#############################################################################

DIRECTIONS_TABLE = {
    "King": [], 
    "Rook": [(1, 0), (0, -1), (0, 1), (-1, 0)],
    "Bishop": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "Queen": [(1, 0), (0, -1), (0, 1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)], 
    "Knight": [], 
    "Ferz": [], 
    "Princess": [(1, 1), (1, -1), (-1, 1), (-1, -1)], 
    "Empress": [(1, 0), (0, -1), (0, 1), (-1, 0)], 
    "Pawn": [] 
}    

JUMPS_TABLE = {
    "King": [(1, 0), (0, -1), (0, 1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)], 
    "Rook": [],
    "Bishop": [],
    "Queen": [], 
    "Knight": [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)], 
    "Ferz": [(1, 1), (1, -1), (-1, 1), (-1, -1)], 
    "Princess": [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)], 
    "Empress": [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)], 
    "Pawn": [(1, 0), (1, 1), (1, -1)] 
} 

PIECE_VALUES = {
    "Pawn": 100, 
    "Knight": 320, 
    "Bishop": 330, 
    "Rook": 500, 
    "Queen": 900, 
    "Ferz": 130, 
    "Princess": 700,
    "Empress": 860,
    "King": 20000
}

class Color(Enum):
    WHITE = auto() 
    BLACK = auto()

def negate(col: Color):
    return Color.BLACK if col is Color.WHITE else Color.WHITE 

@dataclass 
class Piece: 
    """ Represent a piece with a piece_type and a color."""

    piece_type: str 
    color: Color

    def get_directions(self) -> List[Tuple[int]]: 
        return DIRECTIONS_TABLE[self.piece_type] 
    
    def get_jumps(self) -> List[Tuple[int]]: 
        return JUMPS_TABLE[self.piece_type] 

#############################################################################
######## Board
#############################################################################
class Board:    
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.pieces: List[List[Piece]] = [[None for _ in range(cols)] for _ in range(rows)] 
        self.threats_map: Dict[Color, List[List[int]]] = {
            Color.WHITE: [[0 for _ in range(cols)] for _ in range(rows)],
            Color.BLACK: [[0 for _ in range(cols)] for _ in range(rows)]
            }

    def init_pieces(self, pieces): 
        for (row, col), piece in pieces: 
            self.pieces[row][col] = piece 
        for (row, col), _ in pieces: 
            self.process_move(row, col, 1) 

    def increment_threat(self, row: int, col: int, inc: int, color: Color):
        threat_map = self.threats_map[color] 
        threat_map[row][col] += inc 
    
    def place_piece(self, piece: Piece, row: int, col: int) -> None: 
        self.pieces[row][col] = piece 
        self.process_move(row, col, 1) 
    
    def remove_piece(self, row:int, col: int) -> Piece: 
        self.process_move(row, col, -1) 
        temp = self.pieces[row][col] 
        self.pieces[row][col] = None 
        return temp 

    def get_piece(self, row, col) -> Piece:
        return self.pieces[row][col] 
    
    def process_move_directions(self, row, col, mult) -> List[Tuple[int]]:
        rows, cols = ROWS, COLS
        piece = self.get_piece(row, col)
        directions = piece.get_directions() 
        q = deque()
        # Do a BFS in a particular search direction, specific to each piece
        for dx, dy in directions:
            inc = Increment(dx, dy)
            q.append(((row, col), inc))
        
        moves = {}
        while q:
            (nrow, ncol), inc = q.popleft()
            # Check if coordinate out of board boundary
            if nrow < 0 or nrow >= rows or ncol < 0 or ncol >= cols:
                continue 
            if not self.get_piece(nrow, ncol) or self.get_piece(nrow, ncol).color != piece.color: 
                moves[(nrow, ncol)] = True 
            self.increment_threat(nrow, ncol, mult, piece.color) 

            if not self.get_piece(nrow, ncol):    
                nrow, ncol = inc.increment(nrow, ncol)
                q.append(((nrow, ncol), inc)) 
        
        if (row, col) in moves:
            moves.pop((row, col))
        return list(moves.keys()) 
        

    def process_move_jumps(self, row, col, mult) -> List[Tuple[int]]: 
        rows, cols = ROWS, COLS
        piece = self.get_piece(row, col)
        jumps = piece.get_jumps()     
        # Handle black pawn 
        if piece.piece_type == "Pawn" and piece.color is Color.BLACK: 
            jumps = map(lambda d: (-d[0], d[1]), jumps) 
        moves = {}
        for drow, dcol in jumps:
            nrow, ncol = row + drow, col + dcol
            # Check if coordinate out of board boundary
            if nrow < 0 or nrow >= rows or ncol < 0 or ncol >= cols:
                continue

            next_piece = self.get_piece(nrow, ncol)
            if piece.piece_type == "Pawn": 
                can_move_vertical = dcol == 0 and not next_piece
                can_move_diagonal = dcol != 0 and next_piece and next_piece.color != piece.color
                if can_move_vertical or can_move_diagonal:
                    moves[(nrow, ncol)] = True 
            else: 
                empty = not self.get_piece(nrow, ncol)
                can_capture = False if empty else self.get_piece(nrow, ncol).color != piece.color
                if empty or can_capture: 
                    moves[(nrow, ncol)] = True 
            self.increment_threat(nrow, ncol, mult, piece.color) 
        self.increment_threat(row, col, mult, piece.color) 

        if (row, col) in moves:
            moves.pop((row, col))
        return list(moves.keys())

    # Move => Tuple[Tuple[int]] => [((1, 1), (2, 2))]
    def process_move(self, r, c, mult) -> List[Tuple[Tuple[int]]]:
        destinations =  self.process_move_directions(r, c, mult) + self.process_move_jumps(r, c, mult)
        return list(map(lambda d: ((r, c), d), destinations))
    
    def print_board(self):
        mat = [["    " for i in range(COLS)] for j in range(ROWS)]
        for row in range(ROWS):
            for col in range(COLS):
                piece: Piece =  self.pieces[row][col]
                if piece:
                    mat[row][col] =  piece.color + "_" + (piece.piece_type + "     ")[:3] 
        for row in mat:
            print("|".join(row))

def eval_board(board: Board): 
    board 

#############################################################################
######## State
#############################################################################
# Helper class for BFS denoting the direction of the search
class Increment:
    def __init__(self, drow, dcol):
        self.drow, self.dcol = drow, dcol 

    def increment(self, row, col):
        return row + self.drow, col + self.dcol

def to_piece(tup: Tuple[str]):
    piece_type, str_col = tup 
    return Piece(piece_type, Color[str_col.upper()])

@dataclass 
class SavedState: 
    captured_piece: Piece 
    captured_cell: Tuple[int] 
    frm: Tuple[int] 
    to: Tuple[int] 

    def is_captured(self):
        return self.captured_piece != None 
    
class Game:

    def __init__(self, gameboard: Dict[Tuple[int], Tuple[str]]): 
        self.pieces = {Color.WHITE: set(), Color.BLACK: set()}   
        self.turn = Color.WHITE  
        self.board = Board(ROWS, COLS)
        self.prevs: Deque[SavedState] = deque()

        pieces = list(map(
            lambda key: (key, to_piece(gameboard[key])), 
            gameboard
        ))
            
        self.board.init_pieces(pieces) 
        for (row, col), piece in pieces: 
            self.pieces[piece.color].add((row, col)) 
    def switch(self):
        self.turn = negate(self.turn)

    def remove(self, row, col) -> Piece:
        piece = self.board.remove_piece(row, col) 
        self.pieces[self.turn].remove((row, col)) 
        return piece 
    
    def add(self, piece: Piece, row, col):
        self.board.place_piece(piece, row, col) 
        self.pieces[self.turn].add((row, col)) 

    def _move(self, frm: Tuple[int], to: Tuple[int]) -> SavedState: 
        frm_row, frm_col = frm 
        to_row, to_col = to 
        captured = None 

        piece = self.remove(frm_row, frm_col)
        if self.board.get_piece(to_row, to_col):
            self.switch()
            captured = self.remove(to_row, to_col)
            self.switch()
        self.add(piece, to_row, to_col)

        return SavedState(captured, to, frm, to)

    def move(self, frm: Tuple[int], to: Tuple[int]): 
        prev_state = self._move(frm, to) 
        if len(self.prevs) == MAX_DEPTH:
            self.prevs.popleft() 
        self.prevs.append(prev_state) 
        self.switch()

    def undo(self) -> bool : 
        if not self.prevs:
            return False
        self.switch()
        prev_state = self.prevs.pop()
        self._move(prev_state.to, prev_state.frm) 
        if prev_state.is_captured(): 
            self.switch()
            row, col = prev_state.captured_cell[0], prev_state.captured_cell[1]
            self.add(prev_state.captured_piece, row, col) 
            self.switch()
        return True 

    def get_available_moves(self): 
        moves = []
        for row, col in self.pieces[self.turn]: 
            moves.extend(self.board.process_move(row, col, 0))
        return moves 


#Implement your minimax with alpha-beta pruning algorithm here.
def ab(gameboard: Dict[Tuple[int], Tuple[str]]):
    game = Game(gameboard) 
    moves = game.get_available_moves() 


#############################################################################
######## Parser function and helper functions
#############################################################################
### DO NOT EDIT/REMOVE THE FUNCTION BELOW###
# Return number of rows, cols, grid containing obstacles and step costs of coordinates, enemy pieces, own piece, and goal positions
def parse(testcase):
    handle = open(testcase, "r")

    get_par = lambda x: x.split(":")[1]
    rows = int(get_par(handle.readline())) # Integer
    cols = int(get_par(handle.readline())) # Integer
    gameboard = {}
    
    enemy_piece_nums = get_par(handle.readline()).split()
    num_enemy_pieces = 0 # Read Enemy Pieces Positions
    for num in enemy_piece_nums:
        num_enemy_pieces += int(num)

    handle.readline()  # Ignore header
    for i in range(num_enemy_pieces):
        line = handle.readline()[1:-2]
        coords, piece = add_piece(line)
        gameboard[coords] = (piece, "Black")    

    own_piece_nums = get_par(handle.readline()).split()
    num_own_pieces = 0 # Read Own Pieces Positions
    for num in own_piece_nums:
        num_own_pieces += int(num)

    handle.readline()  # Ignore header
    for i in range(num_own_pieces):
        line = handle.readline()[1:-2]
        coords, piece = add_piece(line)
        gameboard[coords] = (piece, "White")    

    return rows, cols, gameboard

def add_piece( comma_seperated) -> Piece:
    piece, ch_coord = comma_seperated.split(",")
    r, c = from_chess_coord(ch_coord)
    return [(r,c), piece]

def from_chess_coord( ch_coord):
    return (int(ch_coord[1:]), ord(ch_coord[0]) - 97)

# You may call this function if you need to set up the board
def main():
    global ROWS, COLS
    config = sys.argv[1]
    rows, cols, gameboard = parse(config)
    ROWS, COLS = rows, cols 
    studentAgent(gameboard)

### DO NOT EDIT/REMOVE THE FUNCTION HEADER BELOW###
# Chess Pieces: King, Queen, Knight, Bishop, Rook, Princess, Empress, Ferz, Pawn (First letter capitalized)
# Colours: White, Black (First Letter capitalized)
# Positions: Tuple. (column (String format), row (Int)). Example: ('a', 0)

# Parameters:
# gameboard: Dictionary of positions (Key) to the tuple of piece type and its colour (Value). This represents the current pieces left on the board.
# Key: position is a tuple with the x-axis in String format and the y-axis in integer format.
# Value: tuple of piece type and piece colour with both values being in String format. Note that the first letter for both type and colour are capitalized as well.
# gameboard example: {('a', 0) : ('Queen', 'White'), ('d', 10) : ('Knight', 'Black'), ('g', 25) : ('Rook', 'White')}
#
# Return value:
# move: A tuple containing the starting position of the piece being moved to the new ending position for the piece. x-axis in String format and y-axis in integer format.
# move example: (('a', 0), ('b', 3))

def studentAgent(gameboard):
    # You can code in here but you cannot remove this function, change its parameter or change the return type
    move = ab(gameboard)
    return move #Format to be returned (('a', 0), ('b', 3))

if __name__ == "__main__": 
    main()