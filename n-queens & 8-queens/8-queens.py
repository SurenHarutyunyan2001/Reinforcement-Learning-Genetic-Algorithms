import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class NQueensProblem:
    # Этот класс инкапсулирует задачу о N-ферзях

    def __init__(self, numOfQueens):
        # param numOfQueens: количество ферзей в задаче
        
        self.numOfQueens = numOfQueens

    def __len__(self):
        # return: количество ферзей
        
        return self.numOfQueens

    def getViolationsCount(self, positions):
        # Вычисляет количество нарушений в данном решении
        # Так как вход содержит уникальные индексы столбцов для каждой строки, 
        # нарушения по строкам и столбцам невозможны. Нужно только подсчитать нарушения на диагоналях.
        # param positions: список индексов, соответствующих позициям ферзей в каждой строке
        # return: вычисленное значение нарушений

        if len(positions) != self.numOfQueens:
            raise ValueError("размер списка позиций должен быть равен ", self.numOfQueens)

        violations = 0

        # итерация по каждой паре ферзей и проверка, находятся ли они на одной диагонали:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):

                # первый ферзь в паре:
                column1 = i
                row1 = positions[i]

                # второй ферзь в паре:
                column2 = j
                row2 = positions[j]

                # проверка на угрозу по диагонали для текущей пары:
                if abs(column1 - column2) == abs(row1 - row2):
                    violations += 1

        return violations

    def plotBoard(self, positions):
        # Рисует позиции ферзей на доске согласно данному решению
        # param positions: список индексов, соответствующих позициям ферзей в каждой строке.
        

        if len(positions) != self.numOfQueens:
            raise ValueError("размер списка позиций должен быть равен ", self.numOfQueens)

        fig, ax = plt.subplots()

        # начинаем с квадратов доски:
        board = np.zeros((self.numOfQueens, self.numOfQueens))
        # меняем цвет каждого второго квадрата:
        board[::2, 1::2] = 1
        board[1::2, ::2] = 1

        # рисуем квадраты с двумя цветами:
        ax.imshow(board, interpolation = 'none', cmap = mpl.colors.ListedColormap(['#ffc794', '#4c2f27']))

        # читаем изображение ферзя и даем ему размер, равный 70% от размера квадрата:
        queenThumbnail = plt.imread('queen-thumbnail.png')
        thumbnailSpread = 0.70 * np.array([-1, 1, -1, 1]) / 2  # spread это [левая, правая, нижняя, верхняя границы]

        # итерация по позициям ферзей - i это строка, j это столбец:
        for i, j in enumerate(positions):
            # размещаем миниатюру ферзя на соответствующем квадрате:
            ax.imshow(queenThumbnail, extent = [j, j, i, i] + thumbnailSpread)

        # отображаем индексы строк и столбцов:
        ax.set(xticks=list(range(self.numOfQueens)), yticks = list(range(self.numOfQueens)))

        ax.axis('image')   # масштабируем график, чтобы он был квадратным

        return plt


# тестирование класса:
def main():
    # создание экземпляра задачи:
    nQueens = NQueensProblem(8)

    # известное правильное решение:
    #solution = [5, 0, 4, 1, 7, 2, 6, 3]

    # решение с 3 нарушениями:
    solution = [1, 2, 7, 5, 0, 3, 4, 6]

    print("Количество нарушений = ", nQueens.getViolationsCount(solution))

    plot = nQueens.plotBoard(solution)
    plot.show()


if __name__ == "__main__":
    main()
