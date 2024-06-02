from typing import Literal
from tabulate import tabulate
import random
import numpy as np

import torch
import torch.nn as nn

# Constants for the game
EMPTY, BLACK, WHITE = 0, 1, -1

square_types = {
    EMPTY: '🟨',
    BLACK: '⚫',
    WHITE: '⚪',
}

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class ReversiDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReversiDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_reversi_board(size=8):
    """Initialize the Reversi board with starting pieces."""
    board = np.zeros((size, size), dtype=int)
    center = size // 2
    board[center - 1][center - 1] = WHITE
    board[center][center] = WHITE
    board[center - 1][center] = BLACK
    board[center][center - 1] = BLACK
    return board

def is_valid_move(board, row, col, player):
    """Check if the move is valid for the player."""
    if board[row][col] != EMPTY:
        return False
    return any(check_direction(board, row, col, player, dr, dc) for dr, dc in DIRECTIONS)

def check_direction(board, row, col, player, dr, dc):
    """Check if there are pieces to flip in a given direction."""
    r, c = row + dr, col + dc
    if not (0 <= r < board.shape[0] and 0 <= c < board.shape[1]) or board[r][c] != -player:
        return False
    while 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
        if board[r][c] == EMPTY:
            return False
        if board[r][c] == player:
            return True
        r += dr
        c += dc
    return False

def apply_move(board, row, col, player):
    """Apply the move and flip the opponent's pieces."""
    board[row][col] = player
    for dr, dc in DIRECTIONS:
        if check_direction(board, row, col, player, dr, dc):
            flip_pieces(board, row, col, player, dr, dc)

def flip_pieces(board, row, col, player, dr, dc):
    """Flip the opponent's pieces in a given direction."""
    r, c = row + dr, col + dc
    while board[r][c] == -player:
        board[r][c] = player
        r += dr
        c += dc

class ReversiGame:
    def __init__(
            self, first_player: Literal['human', 'pc'] = 'human', second_player: Literal['human', 'pc'] = 'pc',
            size=8, output_type='emoji', model_path=None
    ):
        self.size = size
        self.board = build_reversi_board(size)
        if first_player == 'human':
            self.first_player_move = self.human_move
        else:
            self.first_player_move = self.computer_move

        if second_player == 'human':
            self.second_player_move = self.human_move
        else:
            self.second_player_move = self.computer_move

        self.current_player = BLACK
        self.output = output_type

        if model_path:
            self.policy_net = ReversiDQN(3, size * size)
            self.policy_net.load_state_dict(torch.load(model_path))
            self.policy_net.eval()

    def display_board(self):
        if self.output == 'emoji':
            display = [[square_types[cell] for cell in row] for row in self.board]
            print(tabulate(display, showindex=True, headers=[str(i) for i in range(self.size)], tablefmt='plain'))
        else:
            print(self.board)

    def get_valid_moves(self, player):
        return [(r, c) for r in range(self.size) for c in range(self.size) if is_valid_move(self.board, r, c, player)]

    def switch_player(self):
        self.current_player *= -1

    def play(self):
        self.display_board()
        valid_moves = self.get_valid_moves(self.current_player)
        if not valid_moves:
            self.switch_player()
            valid_moves = self.get_valid_moves(self.current_player)
        if valid_moves:
            if self.current_player == BLACK:
                move = self.first_player_move(valid_moves)
            else:
                move = self.second_player_move(valid_moves)
            apply_move(self.board, move[0], move[1], self.current_player)
            self.switch_player()
            return self.play()
        self.show_result()

    def human_move(self, valid_moves):
        print(f"Player {square_types[self.current_player]}'s turn")
        while True:
            move = input("Enter your move (row, col): ")
            try:
                move = tuple(map(int, move.split(',')))
                if move in valid_moves:
                    return move
            except ValueError:
                pass
            print("Invalid move. Try again.")

    def computer_move(self, valid_moves):
        if self.current_player == -1:
            if hasattr(self, 'policy_net'):
                state = self.get_state()
                with torch.no_grad():
                    q_values = self.policy_net(torch.tensor(state).unsqueeze(0))
                    q_values = q_values.flatten()
                    q_values[torch.tensor([r * self.size + c for r in range(self.size) for c in range(self.size) if (r, c) not in valid_moves])] = -float('inf')
                    action = q_values.argmax().item()
                    return divmod(action, self.size)
        return random.choice(valid_moves)

    def get_state(self):
        return np.stack((self.board == BLACK, self.board == WHITE, self.board == EMPTY)).astype(np.float32)

    def get_game_result(self):
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        return {
            "winner": 1 if black_count > white_count else 0 if black_count == white_count else -1,
            "first_player_score": black_count,
            "white_player_score": white_count
        }

    def show_result(self):
        game_result = self.get_game_result()
        winner = game_result['winner']
        if winner == 1:
            print("Black wins!")
        elif winner == -1:
            print("White wins!")
        else:
            print("It's a tie!")
        print(f"Final Score - Black: {game_result['first_player_score']}, White: {game_result['white_player_score']}")

if __name__ == "__main__":
    game = ReversiGame(size=8, output_type='emoji', first_player='pc', model_path="reversi_policy_net.pth")
    game.play()
