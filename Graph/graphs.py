# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class GraphColoringProblem:
    # Этот класс инкапсулирует задачу раскраски графа
    

    def __init__(self, graph, hardConstraintPenalty):
        
        # :param graph: граф NetworkX, который нужно раскрасить
        # :param hardConstraintPenalty: штраф за нарушение жесткого ограничения (нарушение раскраски)
        

        # Инициализация переменных экземпляра:
        self.graph = graph
        self.hardConstraintPenalty = hardConstraintPenalty

        # Список узлов в графе:
        self.nodeList = list(self.graph.nodes)

        # Матрица смежности узлов - 
        # matrix[i,j] равно '1', если узлы i и j соединены, или '0' в противном случае:
        self.adjMatrix = nx.adjacency_matrix(graph).todense()

    def __len__(self):
        
        # :return: количество узлов в графе
        return nx.number_of_nodes(self.graph)

    def getCost(self, colorArrangement):
        
        # Вычисляет стоимость предложенной раскраски
        # :param colorArrangement: список целых чисел, представляющих предложенную раскраску для узлов,
        # один цвет на каждый узел в графе
        # :return: вычисленная стоимость раскраски.
        

        return self.hardConstraintPenalty * self.getViolationsCount(colorArrangement) + self.getNumberOfColors(colorArrangement)

    def getViolationsCount(self, colorArrangement):
        
        # Вычисляет количество нарушений в предложенной раскраске. Каждая пара соединенных узлов
        # с одинаковым цветом считается нарушением.
        # :param colorArrangement: список целых чисел, представляющих предложенную раскраску для узлов,
        # один цвет на каждый узел в графе
        # :return: вычисленное значение
        

        if len(colorArrangement) != self.__len__():
            raise ValueError("Размер раскраски должен быть равен ", self.__len__())

        violations = 0

        # Итерируем по каждой паре узлов и проверяем, соединены ли они И имеют ли одинаковый цвет:
        for i in range(len(colorArrangement)):
            for j in range(i + 1, len(colorArrangement)):

                if self.adjMatrix[i, j]:    # эти узлы соединены
                    if colorArrangement[i] == colorArrangement[j]:
                        violations += 1

        return violations

    def getNumberOfColors(self, colorArrangement):
        
        # Возвращает количество различных цветов в предложенной раскраске
        # :param colorArrangement: список целых чисел, представляющих предложенную раскраску для узлов,
        # один цвет на каждый узел в графе
        # :return: количество различных цветов
        
        return len(set(colorArrangement))

    def plotGraph(self, colorArrangement):
        
        # Строит граф с узлами, окрашенными согласно предложенной раскраске
        # :param colorArrangement: список целых чисел, представляющих предложенную раскраску для узлов,
        # один цвет на каждый узел в графе
        

        if len(colorArrangement) != self.__len__():
            raise ValueError("Размер списка цветов должен быть равен ", self.__len__())

        # Создаем список уникальных цветов в раскраске:
        colorList = list(set(colorArrangement))

        # Создаем соответствующие цвета для целых чисел в списке цветов:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(colorList)))

        # Итерируем по узлам и назначаем каждому соответствующий цвет:
        colorMap = []
        for i in range(self.__len__()):
            color = colors[colorList.index(colorArrangement[i])]
            colorMap.append(color)

        # Строим граф с метками узлов и соответствующими цветами:
        nx.draw_kamada_kawai(self.graph, node_color = colorMap, with_labels = True)
        #nx.draw_circular(self.graph, node_color = color_map, with_labels = True)

        return plt


# Тестирование класса:
def main():
    # Создаем экземпляр задачи с графом Петерсена:
    gcp = GraphColoringProblem(nx.petersen_graph(), 10)

    # Генерируем случайное решение с использованием до 5 различных цветов:
    solution = np.random.randint(5, size = len(gcp))

    print("Решение = ", solution)
    print("Количество цветов = ", gcp.getNumberOfColors(solution))
    print("Количество нарушений = ", gcp.getViolationsCount(solution))
    print("Стоимость = ", gcp.getCost(solution))

    plot = gcp.plotGraph(solution)
    plot.show()


if __name__ == "__main__":
    main()
