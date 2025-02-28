import random
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools

# Константы задачи:
ONE_MAX_LENGTH = 100  # Длина битовой строки, которую нужно оптимизировать

# Константы генетического алгоритма:
POPULATION_SIZE = 200  # Размер популяции
P_CROSSOVER = 0.9  # Вероятность скрещивания
P_MUTATION = 0.1  # Вероятность мутации
MAX_GENERATIONS = 50  # Максимальное число поколений

# Устанавливаем случайное зерно:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Создаем "инструментарий" для генетического алгоритма:
toolbox = base.Toolbox()

# Оператор, который случайным образом возвращает 0 или 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# Определение стратегии максимизации функции приспособленности:
creator.create("FitnessMax", base.Fitness, weights = (1.0,))

# Создание класса Индивидуум на основе списка:
creator.create("Individual", list, fitness = creator.FitnessMax)

# Оператор для создания индивидуумов:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# Оператор для создания популяции:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Функция вычисления приспособленности: 
# Возвращает количество единиц в битовой строке

def oneMaxFitness(individual):
    return sum(individual),  # Возвращаем кортеж

toolbox.register("evaluate", oneMaxFitness)

# Генетические операторы:

# Турнирный отбор с размером турнира 3:
toolbox.register("select", tools.selTournament, tournsize = 3)

# Одноточечное скрещивание:
toolbox.register("mate", tools.cxOnePoint)

# Мутация: инвертирование битов с вероятностью 1/ONE_MAX_LENGTH

toolbox.register("mutate", tools.mutFlipBit, indpb = 1.0 / ONE_MAX_LENGTH)

# Основной поток генетического алгоритма:
def main():
    # Создаем начальную популяцию:
    population = toolbox.populationCreator(n = POPULATION_SIZE)
    generationCounter = 0

    # Вычисляем приспособленность для каждого индивидуума в популяции:
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    # Извлекаем значения приспособленности из популяции:
    fitnessValues = [individual.fitness.values[0] for individual in population]

    # Инициализация списков для хранения статистики:
    maxFitnessValues = []
    meanFitnessValues = []

    # Основной эволюционный цикл:
    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        generationCounter += 1

        # Отбор особей для следующего поколения:
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Применяем оператор скрещивания:
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Применяем оператор мутации:
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Вычисляем приспособленность для новых особей:
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        # Замена текущей популяции на новое поколение:
        population[:] = offspring

        # Обновление статистики:
        fitnessValues = [ind.fitness.values[0] for ind in population]
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print(f"- Поколение {generationCounter}: Макс. приспособленность = {maxFitness}, Средн. приспособленность = {meanFitness}")

        # Вывод лучшего индивидуума:
        best_index = fitnessValues.index(max(fitnessValues))
        print("Лучший индивидуум =", *population[best_index], "\n")

    # Построение графика эволюции:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color = 'red', label = 'Макс. приспособленность')
    plt.plot(meanFitnessValues, color = 'green', label = 'Средняя приспособленность')
    plt.xlabel('Поколение')
    plt.ylabel('Приспособленность')
    plt.title('Эволюция приспособленности')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
