from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

# Длина битовой строки, которую нужно оптимизировать
ONE_MAX_LENGTH = 100

# Константы генетического алгоритма
POPULATION_SIZE = 200  # Размер популяции
P_CROSSOVER = 0.9  # Вероятность кроссовера
P_MUTATION = 0.1   # Вероятность мутации
MAX_GENERATIONS = 50  # Максимальное количество поколений

# Установка зерна для генератора случайных чисел для воспроизводимости
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Создание контейнера для инструментов
toolbox = base.Toolbox()

# Оператор, который случайным образом возвращает 0 или 1
toolbox.register("zeroOrOne", random.randint, 0, 1)

# Определение функции приспособленности (максимизация)
creator.create("FitnessMax", base.Fitness, weights = (1.0,))

# Создание класса "Individual", наследуемого от list
creator.create("Individual", list, fitness=creator.FitnessMax)

# Оператор создания индивидов, заполняющий объект Individual случайными 0 и 1
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# Оператор создания популяции — список индивидов
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Функция оценки приспособленности (считает количество единиц в индивиде)
def oneMaxFitness(individual):
    return sum(individual),  # Возвращаем кортеж

# Регистрация функции оценки
toolbox.register("evaluate", oneMaxFitness)

# Оператор селекции — турнирный отбор (размер турнира 3)
toolbox.register("select", tools.selTournament, tournsize = 3)

# Оператор одноточечного кроссовера
toolbox.register("mate", tools.cxOnePoint)

# Оператор мутации (инверсия битов с вероятностью 1/длина индивида)
toolbox.register("mutate", tools.mutFlipBit, indpb = 1.0 / ONE_MAX_LENGTH)

# Основной цикл работы генетического алгоритма
def main():
    # Создание начальной популяции
    population = toolbox.populationCreator(n = POPULATION_SIZE)

    # Создание объекта для сбора статистики
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # Запуск генетического алгоритма
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb = P_CROSSOVER, mutpb = P_MUTATION, ngen = MAX_GENERATIONS,
        stats = stats, verbose = True
    )

    # Извлечение данных о максимальной и средней приспособленности
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # Визуализация результатов
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color = 'red', label = 'Макс. приспособленность')
    plt.plot(meanFitnessValues, color = 'green', label = 'Сред. приспособленность')
    plt.xlabel('Поколение')
    plt.ylabel('Приспособленность')
    plt.title('Максимальная и средняя приспособленность по поколениям')
    plt.legend()
    plt.show()

# Запуск программы
if __name__ == "__main__":
    main()
