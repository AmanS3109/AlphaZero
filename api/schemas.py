"""Pydantic schemas for the AlphaZero API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request to get raw policy + value from the neural network."""
    game: str = Field(..., description="Game name: 'tictactoe' or 'connect_four'")
    board: list[list[int]] = Field(..., description="2D board array (rows × cols). 1=player1, -1=player2, 0=empty")
    player: int = Field(1, description="Current player (1 or -1)")


class PredictResponse(BaseModel):
    """Raw neural network output."""
    policy: list[float] = Field(..., description="Action probabilities")
    value: float = Field(..., description="Position evaluation in [-1, 1]")


class MoveRequest(BaseModel):
    """Request to get the best AI move via MCTS."""
    game: str = Field(..., description="Game name")
    board: list[list[int]] = Field(..., description="2D board array")
    player: int = Field(1, description="Current player (1 or -1)")
    num_searches: int = Field(100, description="Number of MCTS simulations")


class MoveResponse(BaseModel):
    """MCTS move result."""
    action: int = Field(..., description="Best action index")
    action_probs: list[float] = Field(..., description="Visit-count action probabilities")
    value: float = Field(..., description="Estimated position value")
    new_board: list[list[int]] = Field(..., description="Board after the AI move")
    is_terminal: bool = Field(..., description="Whether the game is over")
    winner: int | None = Field(None, description="Winner (1, -1) or None if draw/not over")


class HealthResponse(BaseModel):
    status: str = "ok"
    games: list[str] = []
