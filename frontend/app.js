/* ══════════════════════════════════════════════════
   AlphaZero Arena — Game Engine & API Integration
   ══════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ── DOM refs ──────────────────────────────────────
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const dom = {
    selectorScreen: $('#selectorScreen'),
    gameScreen: $('#gameScreen'),
    boardContainer: $('#boardContainer'),
    columnIndicators: $('#columnIndicators'),
    statusBar: $('#statusBar'),
    statusIndicator: $('#statusIndicator'),
    statusText: $('#statusText'),
    gameTitle: $('#gameTitle'),
    scoreYou: $('#scoreYou'),
    scoreAI: $('#scoreAI'),
    scoreDraw: $('#scoreDraw'),
    confidenceSection: $('#confidenceSection'),
    confidenceChart: $('#confidenceChart'),
    resultOverlay: $('#resultOverlay'),
    resultEmoji: $('#resultEmoji'),
    resultTitle: $('#resultTitle'),
    resultDesc: $('#resultDesc'),
    btnBack: $('#btnBack'),
    btnNewGame: $('#btnNewGame'),
    btnPlayAgain: $('#btnPlayAgain'),
    difficultySelect: $('#difficultySelect'),
    gameDifficulty: $('#gameDifficulty'),
    bgParticles: $('#bgParticles'),
};

// ── Game state ────────────────────────────────────
const GAMES = {
    tictactoe: {
        name: 'Tic-Tac-Toe',
        rows: 3,
        cols: 3,
        actionSize: 9,
        type: 'tictactoe',
    },
    connect_four: {
        name: 'Connect Four',
        rows: 6,
        cols: 7,
        actionSize: 7,
        type: 'connect_four',
    },
};

let state = {
    game: null,         // game key
    board: null,        // 2D array
    currentPlayer: 1,   // Human = 1, AI = -1
    isThinking: false,
    isTerminal: false,
    scores: { you: 0, ai: 0, draw: 0 },
    numSearches: 100,
};

// ── Init ──────────────────────────────────────────
function init() {
    createParticles();
    loadScores();
    bindEvents();
}

// ── Background particles ─────────────────────────
function createParticles() {
    const colors = ['#22d3ee', '#a78bfa', '#f472b6', '#34d399'];
    for (let i = 0; i < 30; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        const size = Math.random() * 6 + 2;
        const color = colors[Math.floor(Math.random() * colors.length)];
        p.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            left: ${Math.random() * 100}%;
            animation-duration: ${Math.random() * 12 + 10}s;
            animation-delay: ${Math.random() * -20}s;
        `;
        dom.bgParticles.appendChild(p);
    }
}

// ── Score persistence ─────────────────────────────
function loadScores() {
    const saved = sessionStorage.getItem('az_scores');
    if (saved) {
        try { state.scores = JSON.parse(saved); } catch (e) { }
    }
}

function saveScores() {
    sessionStorage.setItem('az_scores', JSON.stringify(state.scores));
}

function updateScoreDisplay() {
    dom.scoreYou.textContent = state.scores.you;
    dom.scoreAI.textContent = state.scores.ai;
    dom.scoreDraw.textContent = `(${state.scores.draw})`;
}

// ── Events ────────────────────────────────────────
function bindEvents() {
    // Game selection
    $$('.game-card').forEach(card => {
        card.addEventListener('click', () => {
            const gameKey = card.dataset.game;
            startGame(gameKey);
        });
    });

    // Difficulty sync
    dom.difficultySelect.addEventListener('change', (e) => {
        state.numSearches = parseInt(e.target.value);
        dom.gameDifficulty.value = e.target.value;
    });
    dom.gameDifficulty.addEventListener('change', (e) => {
        state.numSearches = parseInt(e.target.value);
        dom.difficultySelect.value = e.target.value;
    });

    // Navigation
    dom.btnBack.addEventListener('click', goToSelector);
    dom.btnNewGame.addEventListener('click', () => startGame(state.game));
    dom.btnPlayAgain.addEventListener('click', () => {
        dom.resultOverlay.classList.add('hidden');
        startGame(state.game);
    });
}

// ── Screen transitions ────────────────────────────
function showScreen(screen) {
    $$('.screen').forEach(s => s.classList.remove('active'));
    screen.classList.add('active');
}

function goToSelector() {
    showScreen(dom.selectorScreen);
    dom.confidenceSection.classList.remove('visible');
}

// ── Start a game ──────────────────────────────────
function startGame(gameKey) {
    state.game = gameKey;
    state.currentPlayer = 1;
    state.isThinking = false;
    state.isTerminal = false;

    const cfg = GAMES[gameKey];
    state.board = Array.from({ length: cfg.rows }, () =>
        Array.from({ length: cfg.cols }, () => 0)
    );

    // UI
    dom.gameTitle.textContent = cfg.name;
    dom.confidenceSection.classList.remove('visible');
    dom.resultOverlay.classList.add('hidden');
    updateScoreDisplay();
    setStatus('your-turn', 'Your turn — click to place');
    renderBoard();
    showScreen(dom.gameScreen);

    // Column indicators for Connect Four
    if (gameKey === 'connect_four') {
        dom.columnIndicators.classList.remove('hidden');
        renderColumnIndicators();
    } else {
        dom.columnIndicators.classList.add('hidden');
    }
}

// ── Render board ──────────────────────────────────
function renderBoard() {
    const cfg = GAMES[state.game];
    dom.boardContainer.className = `board-container ${state.game}`;
    dom.boardContainer.innerHTML = '';

    for (let r = 0; r < cfg.rows; r++) {
        for (let c = 0; c < cfg.cols; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = r;
            cell.dataset.col = c;

            const val = state.board[r][c];
            if (val !== 0) {
                cell.classList.add('occupied');
                if (state.game === 'tictactoe') {
                    const piece = document.createElement('span');
                    piece.className = `piece ${val === 1 ? 'x' : 'o'}`;
                    piece.textContent = val === 1 ? '✕' : '◯';
                    cell.appendChild(piece);
                } else {
                    const piece = document.createElement('div');
                    piece.className = `c4-piece ${val === 1 ? 'player1' : 'player2'}`;
                    cell.appendChild(piece);
                }
            }

            if (state.isTerminal || state.isThinking) {
                cell.classList.add('disabled');
            }

            cell.addEventListener('click', () => onCellClick(r, c));
            dom.boardContainer.appendChild(cell);
        }
    }
}

function renderColumnIndicators() {
    dom.columnIndicators.innerHTML = '';
    const cfg = GAMES[state.game];
    for (let c = 0; c < cfg.cols; c++) {
        const ind = document.createElement('div');
        ind.className = 'col-indicator';
        ind.textContent = `▼`;
        // Check if column is full
        if (state.board[0][c] !== 0) {
            ind.classList.add('full');
        }
        ind.addEventListener('click', () => onColumnClick(c));
        dom.columnIndicators.appendChild(ind);
    }
}

// ── Cell click handlers ───────────────────────────
function onCellClick(row, col) {
    if (state.isThinking || state.isTerminal || state.currentPlayer !== 1) return;

    if (state.game === 'tictactoe') {
        if (state.board[row][col] !== 0) return;
        const action = row * GAMES[state.game].cols + col;
        makeHumanMove(action);
    } else {
        // Connect Four — just use the column
        onColumnClick(col);
    }
}

function onColumnClick(col) {
    if (state.isThinking || state.isTerminal || state.currentPlayer !== 1) return;
    if (state.game !== 'connect_four') return;
    // Check column has space
    if (state.board[0][col] !== 0) return;
    makeHumanMove(col);
}

// ── Human move ────────────────────────────────────
function makeHumanMove(action) {
    // Apply locally
    if (state.game === 'tictactoe') {
        const r = Math.floor(action / GAMES[state.game].cols);
        const c = action % GAMES[state.game].cols;
        state.board[r][c] = 1;
    } else {
        // Connect Four — drop to lowest empty row
        for (let r = GAMES[state.game].rows - 1; r >= 0; r--) {
            if (state.board[r][action] === 0) {
                state.board[r][action] = 1;
                break;
            }
        }
    }

    renderBoard();
    if (state.game === 'connect_four') renderColumnIndicators();

    // Check if human won (client-side for instant feedback)
    if (checkTerminal()) return;

    // AI's turn
    state.currentPlayer = -1;
    requestAIMove();
}

// ── AI move via API ───────────────────────────────
async function requestAIMove() {
    state.isThinking = true;
    setStatus('thinking', 'AI is thinking…');
    disableBoard();

    try {
        const resp = await fetch(`${API_BASE}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game: state.game,
                board: state.board,
                player: -1,
                num_searches: state.numSearches,
            }),
        });

        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`API error ${resp.status}: ${err}`);
        }

        const data = await resp.json();

        // Apply AI move
        state.board = data.new_board;
        state.isThinking = false;
        state.currentPlayer = 1;

        renderBoard();
        if (state.game === 'connect_four') renderColumnIndicators();
        showConfidence(data.action_probs, data.action);

        if (data.is_terminal) {
            if (data.winner === -1) {
                endGame('lose');
            } else {
                endGame('draw');
            }
        } else {
            setStatus('your-turn', 'Your turn — click to place');
        }
    } catch (err) {
        state.isThinking = false;
        state.currentPlayer = 1;
        setStatus('error', `Error: ${err.message}`);
        console.error('AI move error:', err);
        enableBoard();
    }
}

// ── Terminal check (client-side) ──────────────────
function checkTerminal() {
    const cfg = GAMES[state.game];

    if (state.game === 'tictactoe') {
        return checkTTTTerminal(cfg);
    } else {
        return checkC4Terminal(cfg);
    }
}

function checkTTTTerminal(cfg) {
    const b = state.board;
    // Rows
    for (let r = 0; r < 3; r++) {
        if (b[r][0] !== 0 && b[r][0] === b[r][1] && b[r][1] === b[r][2]) {
            highlightWin([[r, 0], [r, 1], [r, 2]]);
            endGame(b[r][0] === 1 ? 'win' : 'lose');
            return true;
        }
    }
    // Cols
    for (let c = 0; c < 3; c++) {
        if (b[0][c] !== 0 && b[0][c] === b[1][c] && b[1][c] === b[2][c]) {
            highlightWin([[0, c], [1, c], [2, c]]);
            endGame(b[0][c] === 1 ? 'win' : 'lose');
            return true;
        }
    }
    // Diagonals
    if (b[0][0] !== 0 && b[0][0] === b[1][1] && b[1][1] === b[2][2]) {
        highlightWin([[0, 0], [1, 1], [2, 2]]);
        endGame(b[0][0] === 1 ? 'win' : 'lose');
        return true;
    }
    if (b[0][2] !== 0 && b[0][2] === b[1][1] && b[1][1] === b[2][0]) {
        highlightWin([[0, 2], [1, 1], [2, 0]]);
        endGame(b[0][2] === 1 ? 'win' : 'lose');
        return true;
    }
    // Draw
    const full = b.flat().every(v => v !== 0);
    if (full) {
        endGame('draw');
        return true;
    }
    return false;
}

function checkC4Terminal(cfg) {
    const b = state.board;
    const rows = cfg.rows, cols = cfg.cols;

    // Helper
    const check = (r, c, dr, dc) => {
        const player = b[r][c];
        if (player === 0) return null;
        const cells = [[r, c]];
        for (let i = 1; i < 4; i++) {
            const nr = r + dr * i, nc = c + dc * i;
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols || b[nr][nc] !== player) return null;
            cells.push([nr, nc]);
        }
        return { player, cells };
    };

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
                const result = check(r, c, dr, dc);
                if (result) {
                    highlightWin(result.cells);
                    endGame(result.player === 1 ? 'win' : 'lose');
                    return true;
                }
            }
        }
    }

    // Draw
    const full = b[0].every(v => v !== 0);
    if (full) {
        endGame('draw');
        return true;
    }
    return false;
}

// ── Win highlight ─────────────────────────────────
function highlightWin(cells) {
    cells.forEach(([r, c]) => {
        const cfg = GAMES[state.game];
        const idx = r * cfg.cols + c;
        const cellEl = dom.boardContainer.children[idx];
        if (cellEl) cellEl.classList.add('win-cell');
    });
}

// ── End game ──────────────────────────────────────
function endGame(result) {
    state.isTerminal = true;
    disableBoard();

    const messages = {
        win: { emoji: '🏆', title: 'You Win!', desc: 'Impressive — you outplayed the neural network!', status: 'win' },
        lose: { emoji: '🤖', title: 'AI Wins!', desc: 'The AlphaZero model found the optimal play.', status: 'lose' },
        draw: { emoji: '🤝', title: 'It\'s a Draw', desc: 'Neither side could find a decisive advantage.', status: 'draw' },
    };

    const msg = messages[result];
    if (result === 'win') state.scores.you++;
    if (result === 'lose') state.scores.ai++;
    if (result === 'draw') state.scores.draw++;
    saveScores();
    updateScoreDisplay();

    setStatus(msg.status, msg.title);

    // Show overlay after short delay
    setTimeout(() => {
        dom.resultEmoji.textContent = msg.emoji;
        dom.resultTitle.textContent = msg.title;
        dom.resultDesc.textContent = msg.desc;
        dom.resultOverlay.classList.remove('hidden');
    }, 800);
}

// ── Status helpers ────────────────────────────────
function setStatus(type, text) {
    dom.statusIndicator.className = 'status-indicator';
    if (type === 'thinking') dom.statusIndicator.classList.add('thinking');
    if (type === 'win') dom.statusIndicator.classList.add('win');
    if (type === 'lose') dom.statusIndicator.classList.add('lose');
    if (type === 'draw') dom.statusIndicator.classList.add('draw');
    dom.statusText.textContent = text;
}

function disableBoard() {
    $$('.cell').forEach(c => c.classList.add('disabled'));
}

function enableBoard() {
    $$('.cell').forEach(c => {
        if (!c.classList.contains('occupied')) {
            c.classList.remove('disabled');
        }
    });
}

// ── Confidence chart ──────────────────────────────
function showConfidence(actionProbs, bestAction) {
    dom.confidenceSection.classList.add('visible');
    dom.confidenceChart.innerHTML = '';

    const maxProb = Math.max(...actionProbs);
    const cfg = GAMES[state.game];

    actionProbs.forEach((prob, i) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'conf-bar-wrapper';

        const bar = document.createElement('div');
        bar.className = `conf-bar${i === bestAction ? ' best' : ''}`;
        const heightPct = maxProb > 0 ? (prob / maxProb) * 100 : 0;
        bar.style.height = '0%';
        // Animate in
        requestAnimationFrame(() => {
            setTimeout(() => { bar.style.height = `${heightPct}%`; }, i * 30);
        });

        const label = document.createElement('span');
        label.className = 'conf-label';
        if (state.game === 'tictactoe') {
            const r = Math.floor(i / 3), c = i % 3;
            label.textContent = `${r},${c}`;
        } else {
            label.textContent = `C${i + 1}`;
        }

        const pct = document.createElement('span');
        pct.className = 'conf-pct';
        pct.textContent = `${(prob * 100).toFixed(0)}%`;

        wrapper.appendChild(pct);
        wrapper.appendChild(bar);
        wrapper.appendChild(label);
        dom.confidenceChart.appendChild(wrapper);
    });
}

// ── Boot ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
