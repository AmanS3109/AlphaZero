"""Monte Carlo Tree Search — single and parallel implementations."""

import numpy as np
import torch

from src.games.base import Game
from src.model.resnet import ResNet
from src.mcts.node import Node


class MCTS:
    """
    Standard (single-game) MCTS with neural network guidance.

    Runs `num_searches` simulations from a root state, using the
    ResNet to evaluate leaf nodes and provide prior probabilities.
    """

    def __init__(self, game: Game, args: dict, model: ResNet):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state: np.ndarray) -> np.ndarray:
        """
        Run MCTS from `state` and return action probabilities.

        Args:
            state: Current board state from the current player's perspective.

        Returns:
            Action probability distribution (sum = 1).
        """
        # Define root node
        root = Node(self.game, self.args, state, visit_count=1)

        # Get initial policy from the neural net
        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state(state),
                device=self.model.device,
            ).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

        # Add Dirichlet noise for exploration
        policy = (
            (1 - self.args["dirichlet_epsilon"]) * policy
            + self.args["dirichlet_epsilon"]
            * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size)
        )

        # Mask invalid moves and renormalize
        valid_moves = self.game.get_valid_moves(root.state)
        policy *= valid_moves
        policy /= np.sum(policy)

        root.expand(policy)

        # Run MCTS simulations
        for _ in range(self.args["num_searches"]):
            node = root

            # Selection: descend until we reach a non-expanded node
            while node.is_fully_expanded():
                node = node.select()

            # Get value & terminal flag for this node
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                # Expand & evaluate with the network
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(node.state),
                        device=self.model.device,
                    ).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                value = value.item()

                node.expand(policy)

            # Backpropagate the value up the tree
            node.backpropogate(value)

        # Compute the final action probabilities from visit counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)

        return action_probs


class MCTSParallel:
    """
    Batched MCTS that runs search for multiple games simultaneously.

    Groups leaf nodes across games for a single batched neural network
    forward pass, significantly improving GPU utilization during self-play.
    """

    def __init__(self, game: Game, args: dict, model: ResNet):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, raw_states: np.ndarray, spGames: list) -> None:
        """
        Run parallel MCTS across multiple self-play games.

        Args:
            raw_states: Stacked board states, shape (B, H, W).
            spGames: List of SPG objects holding per-game state.
        """
        B = raw_states.shape[0]

        # Batch-encode each (H, W) board → (B, C, H, W)
        encoded = np.stack([self.game.get_encoded_state(s) for s in raw_states], axis=0)
        t_encoded = torch.tensor(encoded, dtype=torch.float32, device=self.model.device)

        # Initial policy + Dirichlet noise
        logits, _ = self.model(t_encoded)
        policy = torch.softmax(logits, dim=1).cpu().numpy()
        noise = np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * self.game.action_size, size=B
        )
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args["dirichlet_epsilon"] * noise

        # Initialize roots
        for i, spg in enumerate(spGames):
            pm = policy[i]
            valid = self.game.get_valid_moves(raw_states[i])
            pm *= valid
            pm /= pm.sum()
            spg.root = Node(self.game, self.args, raw_states[i], visit_count=1)
            spg.root.expand(pm)

        # MCTS simulations
        for _ in range(self.args["num_searches"]):
            # Selection
            for spg in spGames:
                spg.node = None
                node = spg.root
                while node.is_fully_expanded():
                    node = node.select()

                v, terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                v = self.game.get_opponent_value(v)
                if terminal:
                    node.backpropogate(v)
                else:
                    spg.node = node

            # Expansion + evaluation in batch
            idxs = [i for i, spg in enumerate(spGames) if spg.node is not None]
            if not idxs:
                continue

            new_states = np.stack([spGames[i].node.state for i in idxs], axis=0)
            enc2 = np.stack([self.game.get_encoded_state(s) for s in new_states], axis=0)
            t2 = torch.tensor(enc2, dtype=torch.float32, device=self.model.device)

            logits2, vals2 = self.model(t2)
            policy2 = torch.softmax(logits2, dim=1).cpu().numpy()
            vals = vals2.cpu().numpy().flatten()

            for idx, game_idx in enumerate(idxs):
                node = spGames[game_idx].node
                pm2 = policy2[idx]
                valid = self.game.get_valid_moves(node.state)
                pm2 *= valid
                pm2 /= pm2.sum()
                node.expand(pm2)
                node.backpropogate(vals[idx])
