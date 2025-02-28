# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import numpy as np

class Knapsack01Problem:
    # Этот класс инкапсулирует задачу о рюкзаке 0-1 с сайта RosettaCode.org
    
    def __init__(self):
        # инициализация переменных экземпляра:
        self.items = []
        self.maxCapacity = 0

        # инициализация данных:
        self.__initData()

    def __len__(self):
        # return: общее количество предметов, определённых в задаче
        
        return len(self.items)

    def __initData(self):
        # инициализирует данные задачи о рюкзаке 0-1 с сайта RosettaCode.org
        
        self.items = [
            ("map", 9, 150),
            ("compass", 13, 35),
            ("water", 153, 200),
            ("sandwich", 50, 160),
            ("glucose", 15, 60),
            ("tin", 68, 45),
            ("banana", 27, 60),
            ("apple", 39, 40),
            ("cheese", 23, 30),
            ("beer", 52, 10),
            ("suntan cream", 11, 70),
            ("camera", 32, 30),
            ("t-shirt", 24, 15),
            ("trousers", 48, 10),
            ("umbrella", 73, 40),
            ("waterproof trousers", 42, 70),
            ("waterproof overclothes", 43, 75),
            ("note-case", 22, 80),
            ("sunglasses", 7, 20),
            ("towel", 18, 12),
            ("socks", 4, 50),
            ("book", 30, 10)
        ]

        self.maxCapacity = 400

    def getValue(self, zeroOneList):
        # Рассчитывает стоимость выбранных предметов в списке, игнорируя предметы, которые приведут к превышению максимального веса
        # param zeroOneList: список значений 0/1, соответствующих списку предметов задачи. '1' означает, что предмет выбран.
        # return: рассчитанная стоимость
        
        totalWeight = totalValue = 0

        for i in range(len(zeroOneList)):
            item, weight, value = self.items[i]
            if totalWeight + weight <= self.maxCapacity:
                totalWeight += zeroOneList[i] * weight
                totalValue += zeroOneList[i] * value
        return totalValue

    def printItems(self, zeroOneList):
        # Выводит выбранные предметы из списка, игнорируя предметы, которые приведут к превышению максимального веса
        # param zeroOneList: список значений 0/1, соответствующих списку предметов задачи. '1' означает, что предмет выбран.
        
        totalWeight = totalValue = 0

        for i in range(len(zeroOneList)):
            item, weight, value = self.items[i]
            if totalWeight + weight <= self.maxCapacity:
                if zeroOneList[i] > 0:
                    totalWeight += weight
                    totalValue += value
                    print("- Добавление {}: вес = {}, стоимость = {}, накопленный вес = {}, накопленная стоимость = {}".format(item, weight, value, totalWeight, totalValue))
        print("- Общий вес = {}, Общая стоимость = {}".format(totalWeight, totalValue))


# тестирование класса:
def main():
    # создаем экземпляр задачи:
    knapsack = Knapsack01Problem()

    # создаем случайное решение и оцениваем его:
    randomSolution = np.random.randint(2, size = len(knapsack))
    print("Случайное решение = ")
    print(randomSolution)
    knapsack.printItems(randomSolution)


if __name__ == "__main__":
    main()
