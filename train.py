#!/usr/bin/env python3
"""
CLI entrypoint for AlphaZero training.

Usage:
    python train.py --config configs/default.yaml
    python train.py --game tictactoe --parallel
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

from src.games import get_game
from src.model.resnet import ResNet
from src.training.alphazero import AlphaZero, AlphaZeroParallel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--game", type=str, default=None, help="Override game (tictactoe, connect_four)")
    parser.add_argument("--parallel", action="store_true", help="Use parallel self-play")
    parser.add_argument("--resume", type=str, default=None, help="Path to model checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    game_name = args.game or config.get("game", "tictactoe")

    logger.info(f"Game: {game_name}")
    logger.info(f"Config: {args.config}")

    # Setup
    game = get_game(game_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model_cfg = config.get("model", {})
    model = ResNet(
        game,
        num_resBlocks=model_cfg.get("num_resBlocks", 4),
        num_hidden=model_cfg.get("num_hidden", 64),
        device=device,
    )

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    train_cfg = config.get("training", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    # Build args dict for AlphaZero
    mcts_cfg = config.get("mcts", {})
    az_args = {
        "C": mcts_cfg.get("C", 2),
        "num_searches": mcts_cfg.get("num_searches", 60),
        "dirichlet_epsilon": mcts_cfg.get("dirichlet_epsilon", 0.25),
        "dirichlet_alpha": mcts_cfg.get("dirichlet_alpha", 0.3),
        "num_iterations": train_cfg.get("num_iterations", 3),
        "num_selfPlay_iterations": train_cfg.get("num_selfPlay_iterations", 500),
        "num_parallel_games": train_cfg.get("num_parallel_games", 100),
        "num_epochs": train_cfg.get("num_epochs", 4),
        "batch_size": train_cfg.get("batch_size", 64),
        "temperature": train_cfg.get("temperature", 1.25),
    }

    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints")) / game_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    if args.parallel:
        logger.info("Using AlphaZeroParallel (batched self-play)")
        trainer = AlphaZeroParallel(model, optimizer, game, az_args, str(checkpoint_dir))
    else:
        logger.info("Using AlphaZero (sequential self-play)")
        trainer = AlphaZero(model, optimizer, game, az_args, str(checkpoint_dir))

    start = time.time()
    trainer.learn()
    elapsed = time.time() - start

    logger.info(f"Training complete in {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
