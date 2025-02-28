# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

# Импорт необходимых библиотек
from deap import base, algorithms
from deap import creator
from deap import tools
from matplotlib.lines import Line2D
import random
import matplotlib.pyplot as plt
import numpy as np

# Матрица расстояний между вершинами (D)
inf = 100
D = ((0, 3, 1, 3, inf, inf),
     (3, 0, 4, inf, inf, inf),
     (1, 4, 0, inf, 7, 5),
     (3, inf, inf, 0, inf, 2),
     (inf, inf, 7, inf, 0, 4),
     (inf, inf, 5, 2, 4, 0))

startV = 0              # стартовая вершина
LENGTH_D = len(D)       # количество вершин
LENGTH_CHROM = len(D)*len(D[0])    # длина хромосомы (общее количество возможных связей между вершинами)

# Константы генетического алгоритма
POPULATION_SIZE = 500   # количество индивидуумов в популяции
P_CROSSOVER = 0.9       # вероятность скрещивания
P_MUTATION = 0.1        # вероятность мутации индивидуума
MAX_GENERATIONS = 30    # максимальное количество поколений
HALL_OF_FAME_SIZE = 1   # размер "Зала славы" (для элитизма)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)  # Создание объекта для хранения лучших решений

RANDOM_SEED = 42
random.seed(RANDOM_SEED)  # Установка фиксированного начального состояния для генератора случайных чисел

# Создание структуры для описания задачи минимизации
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))  # Минимизация целевой функции
creator.create("Individual", list, fitness = creator.FitnessMin)  # Создание структуры "Индивид"

toolbox = base.Toolbox()  # Создание инструментария для генетического алгоритма
toolbox.register("randomOrder", random.sample, range(LENGTH_D), LENGTH_D)  # Генерация случайных порядков вершин
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomOrder, LENGTH_D)  # Создание индивидов
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)  # Создание популяции

population = toolbox.populationCreator(n=POPULATION_SIZE)  # Создание начальной популяции

# Функция для вычисления "стоимости" (фитнеса) пути с использованием алгоритма Дейкстры
def dikstryFitness(individual):
    s = 0  # Инициализация стоимости пути
    for n, path in enumerate(individual):  # Для каждого пути
        path = path[:path.index(n) + 1]  # Обработка пути до текущей вершины

        si = startV  # Начальная вершина
        for j in path:  # Пройтись по всем вершинам пути
            s += D[si][j]  # Суммировать расстояния
            si = j  # Перемещаться по пути

    return s,         # Возвращаем стоимость пути как кортеж (для совместимости с DEAP)

# Функция для кроссинговера (перекрестного скрещивания) путей
def cxOrdered(ind1, ind2):
    for p1, p2 in zip(ind1, ind2):
        tools.cxOrdered(p1, p2)  # Применение оператора cxOrdered для обмена порядками

    return ind1, ind2  # Возвращаем измененные индивиды

# Функция для мутации (перемешивания индексов в пути)
def mutShuffleIndexes(individual, indpb):
    for ind in individual:
        tools.mutShuffleIndexes(ind, indpb)  # Перемешивание индексов

    return individual,  # Возвращаем измененный индивид

# Регистрация функций в инструменте
toolbox.register("evaluate", dikstryFitness)  # Регистрация функции для оценки фитнеса
toolbox.register("select", tools.selTournament, tournsize=3)  # Турнирный отбор
toolbox.register("mate", cxOrdered)  # Скрещивание
toolbox.register("mutate", mutShuffleIndexes, indpb = 1.0 / LENGTH_CHROM / 10)  # Мутация

# Статистики: минимальное и среднее значение фитнеса
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

# Выполнение генетического алгоритма
population, logbook = algorithms.eaSimple(population, toolbox,
                                          cxpb = P_CROSSOVER/LENGTH_D,  # Нормализуем вероятность кроссинговера по длине хромосомы
                                          mutpb = P_MUTATION/LENGTH_D,  # Нормализуем вероятность мутации по длине хромосомы
                                          ngen = MAX_GENERATIONS,  # Количество поколений
                                          halloffame = hof,  # Использование Зала славы
                                          stats = stats,  # Статистики
                                          verbose = True)  # Вывод информации в процессе

# Извлечение данных о максимальном и среднем фитнесе
maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")

# Вывод лучшего решения
best = hof.items[0]
print(best)

# Координаты вершин для графика
vertex = ((0, 1), (1, 1), (0.5, 0.8), (0.1, 0.5), (0.8, 0.2), (0.4, 0))

# Список координат для осей X и Y
vx = [v[0] for v in vertex]
vy = [v[1] for v in vertex]

# Функция для отображения графика
def show_graph(ax, best):
    # Отображение линий между вершинами
    ax.add_line(Line2D((vertex[0][0], vertex[1][0]), (vertex[0][1], vertex[1][1]), color = '#aaa'))
    ax.add_line(Line2D((vertex[0][0], vertex[2][0]), (vertex[0][1], vertex[2][1]), color = '#aaa'))
    ax.add_line(Line2D((vertex[0][0], vertex[3][0]), (vertex[0][1], vertex[3][1]), color = '#aaa'))
    ax.add_line(Line2D((vertex[1][0], vertex[2][0]), (vertex[1][1], vertex[2][1]), color = '#aaa'))
    ax.add_line(Line2D((vertex[2][0], vertex[5][0]), (vertex[2][1], vertex[5][1]), color = '#aaa'))
    ax.add_line(Line2D((vertex[2][0], vertex[4][0]), (vertex[2][1], vertex[4][1]), color = '#aaa'))
    ax.add_line(Line2D((vertex[3][0], vertex[5][0]), (vertex[3][1], vertex[5][1]), color = '#aaa'))
    ax.add_line(Line2D((vertex[4][0], vertex[5][0]), (vertex[4][1], vertex[5][1]), color = '#aaa'))

    startV = 0  # Стартовая вершина
    # Отображение пути лучшего индивида
    for i, v in enumerate(best):
        if i == 0:
            continue

        prev = startV
        v = v[: v.index(i) + 1]
        for j in v:
            ax.add_line(Line2D((vertex[prev][0], vertex[j][0]), (vertex[prev][1], vertex[j][1]), color = 'r'))
            prev = j

    ax.plot(vx, vy, ' ob', markersize = 15)  # Отображение вершин

# Построение графика зависимости фитнеса от поколения
plt.plot(maxFitnessValues, color = 'red')
plt.plot(meanFitnessValues, color = 'green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')

# Построение графика маршрута
fig, ax = plt.subplots()
show_graph(ax, best)
plt.show()
