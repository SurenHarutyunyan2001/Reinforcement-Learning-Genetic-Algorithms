# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')
import numpy as np

from deap import base
from deap import creator
from deap import tools
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import elitism
import nurses

# Константы задачи:
HARD_CONSTRAINT_PENALTY = 10  # коэффициент штрафа за нарушение жестких ограничений

# Константы генетического алгоритма:
POPULATION_SIZE = 300
P_CROSSOVER = 0.9  # вероятность кроссовера
P_MUTATION = 0.1   # вероятность мутации индивидуума
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 30

# Устанавливаем начальное значение случайного зерна:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# Создаем экземпляр задачи расписания медсестер:
nsp = nurses.NurseSchedulingProblem(HARD_CONSTRAINT_PENALTY)

# Определяем единственную цель, максимизируя приспособленность:
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))

# Создаем класс Individual на основе списка:
creator.create("Individual", list, fitness = creator.FitnessMin)

# Создаем оператор, который случайным образом возвращает 0 или 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# Создаем оператор для заполнения экземпляра Individual:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(nsp))

# Создаем оператор для популяции, чтобы генерировать список индивидов:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Функция для расчета приспособленности:
def getCost(individual):
    return nsp.getCost(individual),  # возвращаем кортеж

toolbox.register("evaluate", getCost)

# Генетические операторы:
toolbox.register("select", tools.selTournament, tournsize = 2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb = 1.0 / len(nsp))


# Основной процесс генетического алгоритма:
def main():

    # Создаем начальную популяцию (поколение 0):
    population = toolbox.populationCreator(n = POPULATION_SIZE)

    # Подготавливаем объект статистики:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    # Определяем объект hall-of-fame:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Выполняем генетический алгоритм с добавленной функцией hall-of-fame:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb = P_CROSSOVER, mutpb = P_MUTATION,
                                              ngen = MAX_GENERATIONS, stats = stats, halloffame = hof, verbose = True)

    # Печатаем лучшее найденное решение:
    best = hof.items[0]
    print("-- Лучший индивидуум = ", best)
    print("-- Лучшая приспособленность = ", best.fitness.values[0])
    print()
    print("-- Расписание = ")
    nsp.printScheduleInfo(best)

    # Извлекаем статистику:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # Строим график статистики:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color = 'red')
    plt.plot(meanFitnessValues, color = 'green')
    plt.xlabel('Поколение')
    plt.ylabel('Мин / Средняя Приспособленность')
    plt.title('Мин и Средняя приспособленность по поколениям')
    plt.show()


if __name__ == "__main__":
    main()
