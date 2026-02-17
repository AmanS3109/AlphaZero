"""MCTS tree node for AlphaZero search."""

from __future__ import annotations

import math
import numpy as np

from src.games.base import Game


class Node:
    """
    A single node in the Monte Carlo search tree.

    Each node stores the game state, visit statistics, and links to
    parent/children. The UCB formula balances exploitation (Q-value)
    with exploration (prior probability × visit ratio).
    """

    def __init__(
        self,
        game: Game,
        args: dict,
        state: np.ndarray,
        parent: Node | None = None,
        action_taken: int | None = None,
        prior: float = 0.0,
        visit_count: int = 0,
    ):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children: list[Node] = []

        self.visit_count = visit_count
        self.value_sum = 0.0

    def is_fully_expanded(self) -> bool:
        """A node is considered expanded once it has children."""
        return len(self.children) > 0

    def select(self) -> Node:
        """Select the child with the highest UCB score."""
        best_ucb = float("-inf")
        best_child = None

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child  # type: ignore[return-value]

    def get_ucb(self, child: Node) -> float:
        """
        Compute the Upper Confidence Bound for a child node.

        Q-value is normalized to [0, 1] and flipped (opponent's perspective).
        Exploration term uses the PUCT formula with the child's prior.
        """
        if child.visit_count == 0:
            q_value = 0.0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return (
            q_value
            + self.args["C"]
            * (math.sqrt(self.visit_count) / (child.visit_count + 1))
            * child.prior
        )

    def expand(self, policy: np.ndarray) -> None:
        """
        Expand this node by creating a child for each legal action
        weighted by the policy probabilities.
        """
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropogate(self, value: float) -> None:
        """Propagate the evaluation value back up the tree."""
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)

        if self.parent is not None:
            self.parent.backpropogate(value)
