# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import knapsack

# Константы задачи:
# создаем экземпляр задачи о рюкзаке для использования:
knapsack = knapsack.Knapsack01Problem()

# Константы генетического алгоритма:
POPULATION_SIZE = 50
P_CROSSOVER = 0.9  # вероятность кроссовера
P_MUTATION = 0.1   # вероятность мутации индивида
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 1


# Устанавливаем случайное зерно:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# создаем оператор, который случайным образом возвращает 0 или 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# определяем стратегию для одной цели, максимизация приспособленности:
creator.create("FitnessMax", base.Fitness, weights = (1.0,))

# создаем класс Individual на основе списка:
creator.create("Individual", list, fitness = creator.FitnessMax)

# создаем оператор для создания индивидуумов:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(knapsack))

# создаем оператор для генерации популяции (списка индивидов):
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# расчет приспособленности
def knapsackValue(individual):
    return knapsack.getValue(individual),  # возвращаем кортеж


toolbox.register("evaluate", knapsackValue)

# генетические операторы: mutFlipBit

# Турнирный отбор с размером турнира 3:
toolbox.register("select", tools.selTournament, tournsize = 3)

# Кроссовер с одной точкой:
toolbox.register("mate", tools.cxTwoPoint)

# Мутация с переворачиванием бита:
# indpb: независимая вероятность переворота каждого атрибута
toolbox.register("mutate", tools.mutFlipBit, indpb = 1.0 / len(knapsack))


# Поток генетического алгоритма:
def main():

    # создаем начальную популяцию (поколение 0):
    population = toolbox.populationCreator(n = POPULATION_SIZE)

    # подготавливаем объект статистики:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # определяем объект для зала славы:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # выполняем генетический алгоритм с добавлением функции зала славы:
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb = P_CROSSOVER, mutpb = P_MUTATION,
                                              ngen = MAX_GENERATIONS, stats = stats, halloffame = hof, verbose = True)

    # выводим лучшее найденное решение:
    best = hof.items[0]
    print("-- Лучший индивид за все время = ", best)
    print("-- Лучшее приспособление за все время = ", best.fitness.values[0])

    print("-- Предметы для рюкзака = ")
    knapsack.printItems(best)

    # извлекаем статистику:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # строим график статистики:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color = 'red')
    plt.plot(meanFitnessValues, color = 'green')
    plt.xlabel('Поколение')
    plt.ylabel('Макс / Средняя Приспособленность')
    plt.title('Макс и Средняя приспособленность по поколениям')
    plt.show()


if __name__ == "__main__":
    main()
