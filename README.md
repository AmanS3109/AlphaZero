# AlphaZero

A production-level implementation of the [AlphaZero](https://arxiv.org/abs/1712.01815) algorithm for board games, featuring modular game definitions, ResNet-based neural network, Monte Carlo Tree Search (MCTS), and a FastAPI deployment server.

## Supported Games

| Game | Board | Actions | Status |
|------|-------|---------|--------|
| Tic-Tac-Toe | 3×3 | 9 | ✅ Trained |
| Connect Four | 6×7 | 7 | 🔧 Ready to train |

## Project Structure

```
├── src/                    # Core library
│   ├── games/              # Game implementations (TicTacToe, ConnectFour)
│   ├── model/              # ResNet neural network
│   ├── mcts/               # Monte Carlo Tree Search
│   └── training/           # AlphaZero training loop
├── api/                    # FastAPI deployment server
├── configs/                # YAML configuration files
├── checkpoints/            # Trained model weights
├── train.py                # CLI: train a model
├── play.py                 # CLI: play against the AI
├── Dockerfile              # Container deployment
└── notebook/               # Original Jupyter notebook (reference)
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Play Against the AI

Play against the pre-trained TicTacToe model:

```bash
python play.py --game tictactoe --model checkpoints/tictactoe/model_2.pt
```

Options:
- `--num-searches 1000` — More MCTS simulations = stronger play
- `--play-as -1` — Play as O (second player)

### 3. Train a Model

```bash
# Train TicTacToe (sequential, good for debugging)
python train.py --config configs/default.yaml

# Train with parallel self-play (faster, uses GPU better)
python train.py --config configs/default.yaml --parallel

# Train Connect Four
python train.py --game connect_four --config configs/default.yaml --parallel

# Resume from checkpoint
python train.py --resume checkpoints/tictactoe/model_2.pt
```

### 4. Deploy as API

Start the inference server:

```bash
# Development
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Or directly
python -m api.server
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + loaded games |
| `GET` | `/games` | List supported games |
| `POST` | `/predict` | Raw policy + value from neural net |
| `POST` | `/move` | Best move via MCTS search |

#### Example: Get AI Move

```bash
curl -X POST http://localhost:8000/move \
  -H "Content-Type: application/json" \
  -d '{
    "game": "tictactoe",
    "board": [[0,0,0],[0,1,0],[0,0,0]],
    "player": -1,
    "num_searches": 100
  }'
```

Response:
```json
{
  "action": 0,
  "action_probs": [0.45, 0.1, 0.1, ...],
  "value": -0.3,
  "new_board": [[-1,0,0],[0,1,0],[0,0,0]],
  "is_terminal": false,
  "winner": null
}
```

### 5. Docker Deployment

**Build the image:**
```bash
docker build -t alphazero .
```

**Run the API Server:**
```bash
docker run -p 8000:8000 alphazero
```

**Play Directly in Docker:**
To play the interactive CLI game inside the container:
```bash
docker run -it alphazero python play.py --game tictactoe --model checkpoints/tictactoe/model_2.pt
```
*Note: The `-it` flag is required for interactive input.*

## Configuration

Edit `configs/default.yaml` to change hyperparameters:

```yaml
game: tictactoe

model:
  num_resBlocks: 4
  num_hidden: 64

mcts:
  C: 2
  num_searches: 60

training:
  num_iterations: 3
  num_selfPlay_iterations: 500
  num_parallel_games: 100
  num_epochs: 4
  batch_size: 64
```

## Adding a New Game

1. Create `src/games/your_game.py` implementing the `Game` base class
2. Register it in `src/games/__init__.py`
3. Create a config in `configs/`
4. Train: `python train.py --game your_game`

## Architecture

```
Input Board → Encode (3 channels) → ResNet → Policy Head (action probs)
                                           → Value Head (position eval)

MCTS uses the neural network to:
1. Evaluate leaf nodes (value)
2. Guide exploration (policy prior)
3. Select moves (visit count distribution)
```

## License

MIT
