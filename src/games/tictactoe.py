"""Tic-Tac-Toe game implementation for AlphaZero."""

import numpy as np
from src.games.base import Game


class TicTacToe(Game):
    """Standard 3x3 Tic-Tac-Toe. Player 1 = +1, Player 2 = -1."""

    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def __repr__(self) -> str:
        return "TicTacToe"

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        new_state = state.copy()
        row = action // self.column_count
        col = action % self.column_count
        new_state[row, col] = player
        return new_state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray, action: int | None) -> bool:
        if action is None:
            return False
        row = action // self.column_count
        col = action % self.column_count
        player = state[row, col]
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, col]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state: np.ndarray, action: int | None) -> tuple[float, bool]:
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player: int) -> int:
        return -player

    def get_opponent_value(self, value: float) -> float:
        return -value

    def change_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        encoded_state = np.stack([
            (state == -1).astype(np.float32),
            (state == 0).astype(np.float32),
            (state == 1).astype(np.float32),
        ])
        return encoded_state
