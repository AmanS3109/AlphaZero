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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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


def _infer_architecture(state_dict: dict) -> tuple[int, int]:
    """Infer num_resBlocks and num_hidden from a checkpoint state dict."""
    # num_hidden = output channels of the start block conv layer
    num_hidden = state_dict["startBlock.0.weight"].shape[0]
    # num_resBlocks = number of backbone entries (each has conv1.weight)
    num_resblocks = sum(
        1 for k in state_dict if k.endswith(".conv1.weight") and k.startswith("backBone.")
    )
    return num_resblocks, num_hidden


def load_model_for_game(game_name: str) -> ResNet | None:
    """
    Attempt to load the latest checkpoint for a game.
    Looks for model_*.pt files in checkpoints/<game_name>/.
    Auto-detects architecture (resblocks, hidden channels) from the checkpoint.
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

    state_dict = torch.load(latest, map_location=DEVICE)
    num_resblocks, num_hidden = _infer_architecture(state_dict)
    logger.info(f"  Architecture: {num_resblocks} resblocks, {num_hidden} hidden")

    game = get_game(game_name)
    model = ResNet(game, num_resblocks, num_hidden, device=DEVICE)
    model.load_state_dict(state_dict)
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

    # Mount frontend static files (after all API routes are registered)
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        logger.info(f"  ✓ Frontend served from {frontend_dir}")


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


@app.get("/new-game/{game_name}")
def new_game(game_name: str):
    """Return the initial empty board and valid moves for a game."""
    if game_name not in GAME_REGISTRY:
        raise HTTPException(404, f"Unknown game '{game_name}'")
    game = get_game(game_name)
    board = game.get_initial_state()
    valid = game.get_valid_moves(board)
    return {
        "board": board.tolist(),
        "valid_moves": valid.tolist(),
        "rows": game.row_count,
        "cols": game.column_count,
        "action_size": game.action_size,
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
