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
P_CROSSOVER = 0.9  # Вероятность скрещивания
P_MUTATION = 0.1  # Вероятность мутации
MAX_GENERATIONS = 50  # Максимальное количество поколений
HALL_OF_FAME_SIZE = 10  # Размер зала славы

# Устанавливаем случайное зерно для воспроизводимости
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# Оператор, который случайно возвращает 0 или 1

toolbox.register("zeroOrOne", random.randint, 0, 1)

# Определяем стратегию максимизации приспособленности
creator.create("FitnessMax", base.Fitness, weights = (1.0,))

# Создаем класс Individual, наследуемый от list
creator.create("Individual", list, fitness = creator.FitnessMax)

# Создаем операцию генерации индивидов

toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# Создаем операцию генерации популяции

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Функция приспособленности: суммирует количество единиц в особи

def oneMaxFitness(individual):
    return sum(individual),  # Возвращаем кортеж

toolbox.register("evaluate", oneMaxFitness)

# Операторы генетического алгоритма

toolbox.register("select", tools.selTournament, tournsize = 3)  # Турнирный отбор

toolbox.register("mate", tools.cxOnePoint)  # Одноточечное скрещивание

toolbox.register("mutate", tools.mutFlipBit, indpb = 1.0 / ONE_MAX_LENGTH)  # Побитовая мутация

# Основной поток выполнения генетического алгоритма

def main():
    # Создаем начальную популяцию
    population = toolbox.populationCreator(n = POPULATION_SIZE)

    # Создаем объект для сбора статистики
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # Создаем зал славы (Hall of Fame)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Запускаем генетический алгоритм с добавлением зала славы
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb = P_CROSSOVER, mutpb = P_MUTATION,
                                              ngen = MAX_GENERATIONS, stats = stats, halloffame = hof, verbose = True)

    # Выводим информацию о зале славы
    print("Hall of Fame Individuals = ", *hof.items, sep = "\n")
    print("Best Ever Individual = ", hof.items[0])

    # Извлекаем статистику
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # Отображаем статистику в виде графика
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color = 'red')
    plt.plot(meanFitnessValues, color = 'green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()

if __name__ == "__main__":
    main()
