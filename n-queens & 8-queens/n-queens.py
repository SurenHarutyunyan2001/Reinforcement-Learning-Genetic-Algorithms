from deap import base
from deap import creator
from deap import tools
import random
import array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import elitism
import queens

# константы задачи:
NUM_OF_QUEENS = 16

# константы генетического алгоритма:
POPULATION_SIZE = 300
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 30
P_CROSSOVER = 0.9  # вероятность кроссовера
P_MUTATION = 0.1   # вероятность мутации индивидума

# установка случайного зерна для повторяемых результатов
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# создание задачи N-Queens:
nQueens = queens.NQueensProblem(NUM_OF_QUEENS)

toolbox = base.Toolbox()

# определение стратегии минимизации одной цели:
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))

# создание класса Individual, который является списком целых чисел:
creator.create("Individual", array.array, typecode = 'i', fitness = creator.FitnessMin)

# создание оператора для генерации случайных перемешанных индексов:
toolbox.register("randomOrder", random.sample, range(len(nQueens)), len(nQueens))

# создание оператора создания индивидуума с перемешанными индексами:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

# создание оператора для создания популяции:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# расчет приспособленности - вычисление количества нарушений:
def getViolationsCount(individual):
    return nQueens.getViolationsCount(individual),  # возвращает кортеж

toolbox.register("evaluate", getViolationsCount)

# Генетические операторы:
toolbox.register("select", tools.selTournament, tournsize = 2)
toolbox.register("mate", tools.cxUniformPartialyMatched, indpb = 2.0 / len(nQueens))
toolbox.register("mutate", tools.mutShuffleIndexes, indpb = 1.0 / len(nQueens))

# Основной поток генетического алгоритма:
def main():

    # создание начальной популяции (поколение 0):
    population = toolbox.populationCreator(n = POPULATION_SIZE)

    # подготовка объекта статистики:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # создание объекта hall-of-fame:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # выполнение потока генетического алгоритма с добавленным элитизмом:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb = P_CROSSOVER, mutpb = P_MUTATION,
                                              ngen = MAX_GENERATIONS, stats = stats, halloffame = hof, verbose = True)

    # вывод информации о лучших решениях:
    print("- Лучшие решения:")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i].fitness.values[0], " -> ", hof.items[i])

    # построение графиков статистики:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color = 'red')
    plt.plot(meanFitnessValues, color = 'green')
    plt.xlabel('Поколение')
    plt.ylabel('Мин/Средняя приспособленность')
    plt.title('Мин и Средняя приспособленность по поколениям')

    # построение доски для лучшего решения:
    sns.set_style("whitegrid", {'axes.grid' : False})
    nQueens.plotBoard(hof.items[0])

    # отображение обоих графиков:
    plt.show()

if __name__ == "__main__":
    main()
