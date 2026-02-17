#!/usr/bin/env python3
"""
CLI entrypoint for playing against a trained AlphaZero model.

Usage:
    python play.py --game tictactoe --model checkpoints/tictactoe/model_2.pt
    python play.py --game connect_four --model checkpoints/connect_four/model_0_ConnectFour.pt
"""

import argparse
import sys

import numpy as np
import torch

from src.games import get_game
from src.games.base import Game
from src.model.resnet import ResNet
from src.mcts.search import MCTS


# ── Board rendering ────────────────────────────────

SYMBOLS = {1: "X", -1: "O", 0: "·"}


def render_tictactoe(state: np.ndarray) -> str:
    """Render a TicTacToe board as a string."""
    lines = []
    lines.append("  0   1   2")
    for r in range(3):
        row_str = " | ".join(SYMBOLS[state[r, c]] for c in range(3))
        lines.append(f"  {row_str}")
        if r < 2:
            lines.append("  ---------")
    return "\n".join(lines)


def render_connect_four(state: np.ndarray) -> str:
    """Render a Connect Four board as a string."""
    rows, cols = state.shape
    lines = []
    header = "  " + "   ".join(str(c) for c in range(cols))
    lines.append(header)
    for r in range(rows):
        row_str = " | ".join(SYMBOLS[state[r, c]] for c in range(cols))
        lines.append(f"| {row_str} |")
    lines.append("+" + "---+" * cols)
    return "\n".join(lines)


RENDERERS = {
    "TicTacToe": render_tictactoe,
    "ConnectFour": render_connect_four,
}


def render_board(game: Game, state: np.ndarray) -> str:
    renderer = RENDERERS.get(repr(game))
    if renderer:
        return renderer(state)
    return str(state)


# ── Game loop ──────────────────────────────────────

def play_game(game: Game, model: ResNet, args: dict, human_player: int = 1):
    """Interactive human-vs-AI game loop in the terminal."""
    mcts = MCTS(game, args, model)
    state = game.get_initial_state()
    player = 1

    print("\n" + "=" * 40)
    print(f"  AlphaZero — {repr(game)}")
    print(f"  You are {'X' if human_player == 1 else 'O'}")
    print("=" * 40 + "\n")

    while True:
        print(render_board(game, state))
        print()

        valid_moves = game.get_valid_moves(state)

        if player == human_player:
            # Human turn
            while True:
                try:
                    action = int(input(f"Your move ({SYMBOLS[human_player]}) [0-{game.action_size - 1}]: "))
                    if 0 <= action < game.action_size and valid_moves[action]:
                        break
                    print(f"  Invalid move. Valid: {np.where(valid_moves == 1)[0].tolist()}")
                except (ValueError, EOFError):
                    print("  Enter a number.")
        else:
            # AI turn
            print("AI is thinking...")
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = int(np.argmax(mcts_probs))
            print(f"  AI plays: {action}")

        state = game.get_next_state(state, action, player)
        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            print(render_board(game, state))
            print()
            if value == 1:
                winner = SYMBOLS[player]
                if player == human_player:
                    print(f"🎉 You ({winner}) win!")
                else:
                    print(f"🤖 AI ({winner}) wins!")
            else:
                print("🤝 It's a draw!")
            break

        player = game.get_opponent(player)


def main():
    parser = argparse.ArgumentParser(description="Play against AlphaZero")
    parser.add_argument("--game", type=str, required=True, help="Game name (tictactoe, connect_four)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--num-resblocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--num-hidden", type=int, default=64, help="Hidden channels")
    parser.add_argument("--num-searches", type=int, default=1000, help="MCTS simulations per move")
    parser.add_argument("--play-as", type=int, default=1, choices=[1, -1], help="1=X (first), -1=O (second)")
    args = parser.parse_args()

    game = get_game(args.game)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResNet(game, args.num_resblocks, args.num_hidden, device=device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded model: {args.model}")

    mcts_args = {
        "C": 2,
        "num_searches": args.num_searches,
        "dirichlet_epsilon": 0.0,   # No exploration noise during play
        "dirichlet_alpha": 0.03,
    }

    play_game(game, model, mcts_args, human_player=args.play_as)


if __name__ == "__main__":
    main()
