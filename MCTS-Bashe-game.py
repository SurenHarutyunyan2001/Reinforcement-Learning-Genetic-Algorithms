import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import math
import random

class MCTSNode:
    def __init__(self, stones: int, parent: "MCTSNode" = None, move: list[int] = None):
        self.stones: int = stones                                                               # Текущее число камней
        self.parent: MCTSNode = parent                                                          # Родитель 
        self.move: list[int] = move                                                             # Какой ход привел сюда
        self.children: list[MCTSNode] = []                                                      # Дочерние узлы
        self.wins: int = 0                                                                      # Победы
        self.visits: int = 0                                                                    # Кол-во симуляций
        self.untried_moves: list[int] = list(range(1, min(4, stones + 1)))                      # Возможные ходы

    def is_terminal(self) -> bool:
        # Проверка, завершилась ли игра (нет камней).
        return self.stones == 0

    def fully_expanded(self) -> bool:
        # Проверка, все ли ходы из узла уже использованы.
        return len(self.untried_moves) == 0

    def expand(self) -> "MCTSNode":
        # Расширение узла: создаём нового потомка для одного ещё неиспользованного хода.
        move: int = self.untried_moves.pop()    # Дем последний доступный ход
        next_stones: int = self.stones - move
        child: MCTSNode = MCTSNode(next_stones, parent = self, move = move)
        self.children.append(child)
        return child

    def update(self, result: int) -> None:
        # Обновление статистики узла после симуляции.
        # :param result: 1 — победа, 0 — поражение
        self.visits += 1
        self.wins += result

    def best_child(self, c_param: float = 1.4) -> "MCTSNode":
        # Выбор лучшего ребёнка по формуле UCB1.
        # :param c_param: коэффициент баланса между исследованием и использованием

        def ucb1(child: MCTSNode) -> float:
            exploitation = child.wins / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration

        best: MCTSNode = self.children[0]
        best_value: float = ucb1(best)

        for child in self.children[1:]:
            value: float = ucb1(child)
            if value > best_value:
                best = child
                best_value = value

        return best

def simulate_random_game(stones: int) -> int:
    # Симуляция случайной игры до конца.
    player: int = 0           # Начинаем с игрока № 0 (это "текущий игрок").
    while stones > 0:
        move: int = random.randint(1, min(3, stones))
        stones -= move
        player = 1 - player   #  Меняем игрока: если был 0, станет 1 и наоборот.
    return player             #  Возвращаем того, кто не делал последний ход, т.е. победителя.


def mcts(root: MCTSNode, iterations: int) -> int:
    # Основной алгоритм Monte Carlo Tree Search.
    # :param root: корень дерева (текущее состояние)
    # :param iterations: количество симуляций
    # :return: лучший ход (move), по числу посещений
    for _ in range(iterations):
        node: MCTSNode = root

        # 1. Selection
        while not node.is_terminal() and node.fully_expanded():
            node = node.best_child()

        # 2. Expansion
        if not node.is_terminal() and not node.fully_expanded():
            node = node.expand()

        # 3. Simulation
        result_player: int = simulate_random_game(node.stones)

        # 4. Backpropagation
        current: MCTSNode = node
        while current:
            current.update(1 if result_player == 0 else 0)
            result_player = 1 - result_player
            current = current.parent

    # Выбор ребёнка с наибольшим числом посещений
    most_visited: MCTSNode = root.children[0]
    for child in root.children[1:]:
        if child.visits > most_visited.visits:
            most_visited = child
    return most_visited.move if most_visited.move is not None else 1


def play_bache_game(total_stones: int = 10) -> None:
    #Цикл игры: MCTS против случайного игрока.
    #:param total_stones: стартовое количество камней

    player: int = 0  # 0 - MCTS, 1 - случайный игрок
    stones: int = total_stones
    turn: int = 1
    while stones > 0:
        print(f"\nХод {turn}. Осталось камней: {stones}")
        if player == 0:
            root: MCTSNode = MCTSNode(stones)
            move: int = mcts(root, 1000)
            print(f"MCTS берёт {move}")
        else:
            move: int = random.randint(1, min(3, stones))
            print(f"Случайный игрок берёт {move}")
        stones -= move
        player = 1 - player
        turn += 1
    winner: str = "MCTS" if player == 1 else "Случайный игрок"
    print(f"\nПобедил: {winner}")


play_bache_game(21)
   
