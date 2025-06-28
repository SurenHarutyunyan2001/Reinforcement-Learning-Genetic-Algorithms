import random
from typing import List, Optional, Tuple

# Представление состояния игры
class TicTacToe:
    def __init__(self, board: Optional[List[List[str]]] = None, player: str = "X"):
        self.board: List[List[str]] = board if board else [[" "]*3 for _ in range(3)]
        self.player: str = player  # "X" или "O"

    def clone(self) -> "TicTacToe":
        return TicTacToe([row.copy() for row in self.board], self.player)

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == " "]

    def make_move(self, move: Tuple[int, int]) -> None:
        i, j = move
        if self.board[i][j] == " ":
            self.board[i][j] = self.player
            self.player = "O" if self.player == "X" else "X"

    def is_game_over(self) -> bool:
        return self.get_winner() is not None or not self.get_legal_moves()

    def get_winner(self) -> Optional[str]:
        lines = self.board + \
                [[self.board[i][j] for i in range(3)] for j in range(3)] + \
                [[self.board[i][i] for i in range(3)], [self.board[i][2-i] for i in range(3)]]
        for line in lines:
            if line[0] != " " and all(cell == line[0] for cell in line):
                return line[0]
        return None

class MCTSNode:
    def __init__(self, state: TicTacToe, parent: Optional["MCTSNode"] = None):
        self.state: TicTacToe = state
        self.parent: Optional["MCTSNode"] = parent
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.wins: int = 0
        self.untried_moves: List[Tuple[int, int]] = state.get_legal_moves()

    def expand(self) -> "MCTSNode":
        move: Tuple[int, int] = self.untried_moves.pop()
        new_state = self.state.clone()
        new_state.make_move(move)
        child = MCTSNode(new_state, self)
        self.children.append(child)
        return child

    def is_fully_expanded(self) -> bool:
        return not self.untried_moves

    def best_child(self, c_param: float = 1.41) -> "MCTSNode":
        def ucb_score(node: MCTSNode) -> float:
            if node.visits == 0:
                return float("inf")
            return (node.wins / node.visits) + c_param * ((2 * (self.visits)**0.5) / node.visits)
        return max(self.children, key=ucb_score)

    def update(self, result: int) -> None:
        self.visits += 1
        self.wins += result

def simulate_random_game(state: TicTacToe) -> Optional[str]:
    simulation = state.clone()
    while not simulation.is_game_over():
        moves = simulation.get_legal_moves()
        simulation.make_move(random.choice(moves))
    return simulation.get_winner()

def backpropagate(node: MCTSNode, winner: Optional[str]) -> None:
    current: Optional[MCTSNode] = node
    result_player = 0 if winner == "X" else 1 if winner == "O" else -1

    while current is not None:
        if result_player != -1:
            current.update(1 if result_player == 0 else 0)
            result_player = 1 - result_player  # Переключаем игрока
        current = current.parent


def mcts(root_state: TicTacToe, iterations: int = 1000) -> Tuple[int, int]:
    root: MCTSNode = MCTSNode(root_state)

    for _ in range(iterations):
        node: MCTSNode = root

        # 1. Selection
        while not node.state.is_game_over() and node.is_fully_expanded():
            node = node.best_child()

        # 2. Expansion
        if not node.state.is_game_over() and not node.is_fully_expanded():
            node = node.expand()

        # 3. Simulation
        winner: Optional[str] = simulate_random_game(node.state)

        # 4. Backpropagation
        backpropagate(node, winner)

    # Возвращаем лучший ход
    best_move = max(root.children, key=lambda n: n.visits)
    for i in range(3):
        for j in range(3):
            if root_state.board[i][j] != best_move.state.board[i][j]:
                return i, j
    return -1, -1  # fallback

def play_tictactoe():
    game = TicTacToe()
    while not game.is_game_over():
        print_board(game.board)
        if game.player == "X":
            move = mcts(game, 500)
        else:
            move = random.choice(game.get_legal_moves())
        game.make_move(move)

    print_board(game.board)
    winner = game.get_winner()
    print("Победитель:", winner if winner else "Ничья")

def print_board(board: List[List[str]]) -> None:
    print("\n".join([" | ".join(row) for row in board]))
    print("---" * 3)

# Запуск игры
play_tictactoe()
