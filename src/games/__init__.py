from src.games.tictactoe import TicTacToe
from src.games.connect_four import ConnectFour

GAME_REGISTRY = {
    "tictactoe": TicTacToe,
    "connect_four": ConnectFour,
}


def get_game(name: str):
    """Get a game instance by name."""
    if name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game '{name}'. Available: {list(GAME_REGISTRY.keys())}")
    return GAME_REGISTRY[name]()
