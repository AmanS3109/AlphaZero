"""Abstract base class for all AlphaZero-compatible games."""

from abc import ABC, abstractmethod
import numpy as np


class Game(ABC):
    """
    Interface that any game must implement to work with the AlphaZero
    training pipeline and MCTS search.
    """

    row_count: int
    column_count: int
    action_size: int

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Return the empty starting board as a numpy array."""
        ...

    @abstractmethod
    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """Return a new state after `player` takes `action`."""
        ...

    @abstractmethod
    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """Return a binary mask of valid actions (1=legal, 0=illegal)."""
        ...

    @abstractmethod
    def check_win(self, state: np.ndarray, action: int | None) -> bool:
        """Check if the last `action` resulted in a win."""
        ...

    @abstractmethod
    def get_value_and_terminated(self, state: np.ndarray, action: int | None) -> tuple[float, bool]:
        """
        Return (value, is_terminal).
        value=1 means the player who just moved won; 0 means draw.
        """
        ...

    @abstractmethod
    def get_opponent(self, player: int) -> int:
        """Return the opponent of `player`."""
        ...

    @abstractmethod
    def get_opponent_value(self, value: float) -> float:
        """Flip the value to the opponent's perspective."""
        ...

    @abstractmethod
    def change_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        """Return the state from `player`'s perspective."""
        ...

    @abstractmethod
    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        """
        Encode the state into a (C, H, W) tensor suitable for the neural network.
        Typically 3 channels: opponent pieces, empty squares, current player pieces.
        """
        ...
