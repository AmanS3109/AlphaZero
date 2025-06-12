# 🧠 AlphaZero from Scratch

A from-scratch implementation of AlphaZero, showcasing powerful tree search and deep learning techniques for playing board games like **TicTacToe** and **Connect Four**. Built with **PyTorch**, **NumPy**, and custom UIs, this project explores and compares different variants of **Monte Carlo Tree Search (MCTS)** — including **AlphaMCTS** and their **parallelized versions** — with a focus on **speed optimization** and **strategic play**.

---

## 🚀 Features

- ✅ **Pure Python implementation** (no external RL libraries)
- ♟️ **Two classic games**: TicTacToe and Connect Four
- 🧮 **Algorithms implemented**:
  - Standard MCTS
  - AlphaMCTS (neural-guided MCTS)
  - MCTSParallel
  - AlphaMCTSParallel
- ⚡ **Optimized for speed** using multiprocessing in Python
- 📊 **Matplotlib visualizations** of training statistics
- 🤖 **Evaluation using Kaggle Environments** (agents compete against each other)
- 🖼️ Simple **Python-based UI** to play against trained agents

---

## 🧱 Tech Stack

- **Python**
- **PyTorch**
- **NumPy**
- **Matplotlib**
- **Kaggle Environments**

---

## 📂 Project Structure

```bash
.
├── main.ipynb           # All code: game logic, MCTS, model, training, UI
├── requirements.txt     # Dependencies
└── README.md
