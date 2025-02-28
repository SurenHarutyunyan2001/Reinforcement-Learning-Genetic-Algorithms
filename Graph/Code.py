# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from deap import base
from deap import creator
from deap import tools
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import elitism
import graphs


# Константы задачи:
HARD_CONSTRAINT_PENALTY = 10  # штрафной коэффициент за нарушение жесткого ограничения

# Константы генетического алгоритма:
POPULATION_SIZE = 100
P_CROSSOVER = 0.9  # вероятность кроссинговера
P_MUTATION = 0.1   # вероятность мутации индивида
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 5
MAX_COLORS = 10

# Устанавливаем случайное начальное значение:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# Создаем экземпляр задачи раскраски графа, который будет использоваться:
gcp = graphs.GraphColoringProblem(nx.petersen_graph(), HARD_CONSTRAINT_PENALTY)
#gcp = graphs.GraphColoringProblem(nx.mycielski_graph(5), HARD_CONSTRAINT_PENALTY)

# Определяем одну цель: максимизация приспособленности:
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))

# Создаем класс Individual на основе списка:
creator.create("Individual", list, fitness=creator.FitnessMin)

# Создаем оператор, который случайным образом возвращает целое число в диапазоне допустимых цветов:
toolbox.register("Integers", random.randint, 0, MAX_COLORS - 1)

# Создаем оператор для создания индивидов, заполняющих экземпляр Individual:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Integers, len(gcp))

# Создаем оператор для создания популяции, генерирующей список индивидов:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# Вычисление приспособленности: стоимость предложенного решения
def getCost(individual):
    return gcp.getCost(individual),  # возвращаем кортеж


toolbox.register("evaluate", getCost)

# Генетические операторы:
toolbox.register("select", tools.selTournament, tournsize = 2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low = 0, up = MAX_COLORS - 1, indpb = 1.0 / len(gcp))


# Поток генетического алгоритма:
def main():

    # Создаем начальную популяцию (поколение 0):
    population = toolbox.populationCreator(n = POPULATION_SIZE)

    # Подготавливаем объект статистики:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    # Определяем объект зала славы:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Выполняем генетический алгоритм с элитизмом:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb = P_CROSSOVER, mutpb = P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats = stats, halloffame = hof, verbose = True)

    # Выводим информацию о лучшем решении:
    best = hof.items[0]
    print("-- Лучший индивид = ", best)
    print("-- Лучшая приспособленность = ", best.fitness.values[0])
    print()
    print("Количество цветов = ", gcp.getNumberOfColors(best))
    print("Количество нарушений = ", gcp.getViolationsCount(best))
    print("Стоимость = ", gcp.getCost(best))

    # Строим график для лучшего решения:
    plt.figure(1)
    gcp.plotGraph(best)

    # Извлекаем статистику:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # Строим график статистики:
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color = 'red')
    plt.plot(meanFitnessValues, color = 'green')
    plt.xlabel('Поколение')
    plt.ylabel('Мин / Средняя приспособленность')
    plt.title('Мин и Средняя приспособленность за поколения')

    plt.show()


if __name__ == "__main__":
    main()
