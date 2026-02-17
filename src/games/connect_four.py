"""Connect Four game implementation for AlphaZero."""

import numpy as np
from src.games.base import Game


class ConnectFour(Game):
    """Standard 6×7 Connect Four. Player 1 = +1, Player 2 = -1."""

    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def __repr__(self) -> str:
        return "ConnectFour"

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        new_state = state.copy()
        row = np.max(np.where(state[:, action] == 0))
        new_state[row, action] = player
        return new_state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray, action: int | None) -> bool:
        if action is None:
            return False
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row: int, offset_column: int) -> int:
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1  # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1  # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1  # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1  # top right diagonal
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
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
