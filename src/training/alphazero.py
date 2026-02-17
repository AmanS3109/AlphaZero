"""AlphaZero self-play and training loop — sequential and parallel."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from src.games.base import Game
from src.model.resnet import ResNet
from src.mcts.search import MCTS, MCTSParallel

logger = logging.getLogger(__name__)


class SPG:
    """Self-Play Game container — holds state and memory for one active game."""

    def __init__(self, game: Game):
        self.state = game.get_initial_state()
        self.memory: list[tuple] = []
        self.root = None
        self.node = None


class AlphaZero:
    """
    Sequential AlphaZero training loop.

    One game of self-play is run at a time. Suitable for debugging
    and small games. For better GPU utilization, use AlphaZeroParallel.
    """

    def __init__(
        self,
        model: ResNet,
        optimizer: torch.optim.Optimizer,
        game: Game,
        args: dict,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def selfPlay(self) -> list[tuple]:
        """Play one full game via MCTS self-play and return training samples."""
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args["temperature"])
            temperature_action_probs /= temperature_action_probs.sum()

            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)
            if is_terminal:
                break

            player = self.game.get_opponent(player)

        # Package training samples with the final reward
        training_data = []
        for hist_state, hist_pi, hist_player in memory:
            outcome = value if hist_player == player else self.game.get_opponent_value(value)
            training_data.append((
                self.game.get_encoded_state(hist_state),
                hist_pi,
                outcome,
            ))
        return training_data

    def train(self, memory: list[tuple]) -> None:
        """Train the model on collected self-play data."""
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx : min(len(memory) - 1, batchIdx + self.args["batch_size"])]
            if not sample:
                continue
            state, policy_targets, value_targets = zip(*sample)

            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(
                np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device
            )

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self) -> None:
        """Full AlphaZero training loop: self-play → train → checkpoint."""
        for iteration in range(self.args["num_iterations"]):
            logger.info(f"Iteration {iteration + 1}/{self.args['num_iterations']}")

            # Self-play
            memory = []
            self.model.eval()
            for sp_iter in trange(self.args["num_selfPlay_iterations"], desc="Self-play"):
                memory += self.selfPlay()
            logger.info(f"  Collected {len(memory)} training samples")

            # Train
            self.model.train()
            for epoch in trange(self.args["num_epochs"], desc="Training"):
                self.train(memory)

            # Save checkpoints
            model_path = self.checkpoint_dir / f"model_{iteration}_{self.game}.pt"
            optim_path = self.checkpoint_dir / f"optim_{iteration}_{self.game}.pt"
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optim_path)
            logger.info(f"  Saved checkpoint: {model_path}")


class AlphaZeroParallel:
    """
    Parallel AlphaZero training loop.

    Runs multiple self-play games simultaneously, batching neural
    network evaluations for significantly better GPU utilization.
    """

    def __init__(
        self,
        model: ResNet,
        optimizer: torch.optim.Optimizer,
        game: Game,
        args: dict,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def selfPlay(self) -> list[tuple]:
        """Run parallel self-play games and return training samples."""
        memory = []
        player = 1
        spGames = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]

        while spGames:
            raw_states = np.stack([spg.state for spg in spGames])
            raw_states = self.game.change_perspective(raw_states, player)

            # Parallel MCTS
            self.mcts.search(raw_states, spGames)

            # Advance each game
            for i in range(len(spGames) - 1, -1, -1):
                spg = spGames[i]
                # Visit-count → action_probs
                ap = np.zeros(self.game.action_size)
                for c in spg.root.children:
                    ap[c.action_taken] = c.visit_count
                ap /= ap.sum()
                spg.memory.append((spg.root.state, ap, player))

                # Sample with temperature
                tp = ap ** (1 / self.args["temperature"])
                tp /= tp.sum()
                a = np.random.choice(self.game.action_size, p=tp)

                spg.state = self.game.get_next_state(spg.state, a, player)
                v, done = self.game.get_value_and_terminated(spg.state, a)

                if done:
                    for hs, hap, hp in spg.memory:
                        outcome = v if hp == player else self.game.get_opponent_value(v)
                        memory.append((self.game.get_encoded_state(hs), hap, outcome))
                    spGames.pop(i)

            player = self.game.get_opponent(player)

        return memory

    def train(self, memory: list[tuple]) -> None:
        """Train the model on collected self-play data."""
        random.shuffle(memory)
        for start in range(0, len(memory), self.args["batch_size"]):
            batch = memory[start : start + self.args["batch_size"]]
            if not batch:
                continue
            states, pols, vals = zip(*batch)

            s = torch.tensor(np.array(states), dtype=torch.float32, device=self.model.device)
            p = torch.tensor(np.array(pols), dtype=torch.float32, device=self.model.device)
            v = torch.tensor(
                np.array(vals).reshape(-1, 1), dtype=torch.float32, device=self.model.device
            )

            out_p, out_v = self.model(s)
            loss = F.cross_entropy(out_p, p) + F.mse_loss(out_v, v)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self) -> None:
        """Full parallel AlphaZero training loop."""
        for it in range(self.args["num_iterations"]):
            logger.info(f"Iteration {it + 1}/{self.args['num_iterations']}")

            # Self-play
            self.model.eval()
            mem = []
            n = self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"]
            for _ in trange(n, desc="Self-play"):
                mem += self.selfPlay()
            logger.info(f"  Collected {len(mem)} training samples")

            # Training
            self.model.train()
            for _ in trange(self.args["num_epochs"], desc="Training"):
                self.train(mem)

            # Save checkpoints
            cls = self.game.__class__.__name__
            model_path = self.checkpoint_dir / f"model_{it}_{cls}.pt"
            optim_path = self.checkpoint_dir / f"optim_{it}_{cls}.pt"
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optim_path)
            logger.info(f"  Saved checkpoint: {model_path}")
