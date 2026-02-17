"""
FastAPI server for deploying trained AlphaZero models.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Or:
    python -m api.server
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    MoveRequest,
    MoveResponse,
    PredictRequest,
    PredictResponse,
)
from src.games import GAME_REGISTRY, get_game
from src.model.resnet import ResNet
from src.mcts.search import MCTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ── App ──────────────────────────────────────────

app = FastAPI(
    title="AlphaZero API",
    description="Serve trained AlphaZero models for game AI inference",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model registry ───────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: dict[str, ResNet] = {}  # game_name → loaded model

# Config via env vars (or defaults)
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
NUM_RESBLOCKS = int(os.environ.get("NUM_RESBLOCKS", "4"))
NUM_HIDDEN = int(os.environ.get("NUM_HIDDEN", "64"))


def load_model_for_game(game_name: str) -> ResNet | None:
    """
    Attempt to load the latest checkpoint for a game.
    Looks for model_*.pt files in checkpoints/<game_name>/.
    """
    ckpt_dir = Path(CHECKPOINT_DIR) / game_name
    if not ckpt_dir.exists():
        return None

    # Find the highest-numbered model checkpoint
    model_files = sorted(ckpt_dir.glob("model_*.pt"))
    if not model_files:
        return None

    latest = model_files[-1]
    logger.info(f"Loading {game_name} model from {latest}")

    game = get_game(game_name)
    model = ResNet(game, NUM_RESBLOCKS, NUM_HIDDEN, device=DEVICE)
    model.load_state_dict(torch.load(latest, map_location=DEVICE))
    model.eval()
    return model


@app.on_event("startup")
def startup():
    """Load all available models on server start."""
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Checkpoint dir: {CHECKPOINT_DIR}")

    for game_name in GAME_REGISTRY:
        model = load_model_for_game(game_name)
        if model is not None:
            MODELS[game_name] = model
            logger.info(f"  ✓ {game_name} model loaded")
        else:
            logger.info(f"  ✗ {game_name} — no checkpoint found")

    if not MODELS:
        logger.warning("No models loaded! API will return errors for /predict and /move.")


# ── Endpoints ────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", games=list(MODELS.keys()))


@app.get("/games")
def list_games():
    """List all supported games and their loading status."""
    return {
        name: {"loaded": name in MODELS}
        for name in GAME_REGISTRY
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Get raw neural network policy + value for a board position.
    No MCTS search is performed.
    """
    if req.game not in MODELS:
        raise HTTPException(404, f"No model loaded for '{req.game}'")

    game = get_game(req.game)
    model = MODELS[req.game]

    board = np.array(req.board, dtype=int)
    if req.player != 1:
        board = game.change_perspective(board, req.player)

    encoded = game.get_encoded_state(board)
    tensor = torch.tensor(encoded, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = model(tensor)

    policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
    return PredictResponse(
        policy=policy.tolist(),
        value=float(value.item()),
    )


@app.post("/move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    """
    Run MCTS and return the best move for the current player.
    Also returns the new board state after the move.
    """
    if req.game not in MODELS:
        raise HTTPException(404, f"No model loaded for '{req.game}'")

    game = get_game(req.game)
    model = MODELS[req.game]

    board = np.array(req.board, dtype=int)
    neutral_board = game.change_perspective(board, req.player)

    mcts_args = {
        "C": 2,
        "num_searches": req.num_searches,
        "dirichlet_epsilon": 0.0,   # No exploration noise for play
        "dirichlet_alpha": 0.03,
    }

    mcts = MCTS(game, mcts_args, model)
    action_probs = mcts.search(neutral_board)
    action = int(np.argmax(action_probs))

    # Apply the move
    new_board = game.get_next_state(board, action, req.player)

    # Check terminal
    value, is_terminal = game.get_value_and_terminated(new_board, action)
    winner = None
    if is_terminal and value == 1:
        winner = req.player

    return MoveResponse(
        action=action,
        action_probs=action_probs.tolist(),
        value=float(value),
        new_board=new_board.tolist(),
        is_terminal=is_terminal,
        winner=winner,
    )


# ── Run directly ─────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
