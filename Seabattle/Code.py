# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from deap import base, algorithms
from deap import creator
from deap import tools
import algelitism
from graph_show import show_graph, show_ships
import random
import matplotlib.pyplot as plt
import numpy as np

# Константы, определяющие размеры поля и количество кораблей
POLE_SIZE = 7
SHIPS = 10
LENGTH_CHROM = 3 * SHIPS  # Длина хромосомы, подлежащей оптимизации (каждый корабль описывается тремя числами)

# Параметры генетического алгоритма
POPULATION_SIZE = 200   # Количество индивидуумов в популяции
P_CROSSOVER = 0.9       # Вероятность скрещивания
P_MUTATION = 0.2        # Вероятность мутации
MAX_GENERATIONS = 50    # Максимальное количество поколений
HALL_OF_FAME_SIZE = 1   # Количество лучших индивидуумов, сохраняемых в Hall of Fame

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)  # Создание объекта для хранения лучших индивидуумов

RANDOM_SEED = 42
random.seed(RANDOM_SEED)  # Инициализация генератора случайных чисел для воспроизводимости

# Создание классов для оценки и создания индивидуумов
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))  # Минлинг задача (меньше приспособленность - лучше)
creator.create("Individual", list, fitness = creator.FitnessMin)  # Индивидуум будет представлять собой список

# Функция для случайного создания хромосомы
def randomShip(total):
    ships = []
    for n in range(total):
        # Каждое судно представляется тремя числами (позиция X, позиция Y, ориентация)
        ships.extend([random.randint(1, POLE_SIZE), random.randint(1, POLE_SIZE), random.randint(0, 1)])
    return creator.Individual(ships)

# Инициализация инструментария для генетического алгоритма
toolbox = base.Toolbox()
toolbox.register("randomShip", randomShip, SHIPS)  # Регистрация функции для создания корабля
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.randomShip)  # Регистрация функции для создания популяции

# Создание начальной популяции
population = toolbox.populationCreator(n=POPULATION_SIZE)

# Функция оценки приспособленности для каждого индивидуума (т.е. корабля)
def shipsFitness(individual):
    # Типы кораблей: 4, 3, 3, 2, 2, 2, 1, 1, 1, 1
    type_ship = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]

    inf = 1000
    P0 = np.zeros((POLE_SIZE, POLE_SIZE))  # Пустое поле
    P = np.ones((POLE_SIZE + 6, POLE_SIZE + 6)) * inf  # Рабочее поле с запасом
    P[1:POLE_SIZE + 1, 1 : POLE_SIZE + 1] = P0  # Заполнение области, соответствующей полю

    th = 0.2
    h = np.ones((3, 6)) * th  # Для горизонтальных кораблей
    ship_one = np.ones((1, 4))  # Единичная длина корабля
    v = np.ones((6, 3)) * th  # Для вертикальных кораблей

    # Размещение кораблей на поле
    for *ship, t in zip(*[iter(individual)] * 3, type_ship):
        if ship[-1] == 0:  # Горизонтальный корабль
            sh = np.copy(h[:, : t + 2])  # Создаем копию шаблона
            sh[1, 1 :t + 1] = ship_one[0, :t]  # Размещение корабля на поле
            P[ship[0] - 1 :ship[0] + 2, ship[1] - 1 :ship[1] + t + 1] += sh  # Размещение на поле
        else:  # Вертикальный корабль
            sh = np.copy(v[:t + 2, :])  # Копируем шаблон
            sh[1 :t + 1, 1] = ship_one[0, :t]  # Размещение корабля
            P[ship[0]-1 :ship[0] + t + 1, ship[1] - 1 :ship[1] + 2] += sh  # Размещение на поле

    # Оценка приспособленности - сумма значений на поле
    s = np.sum(P[np.bitwise_and(P > 1, P < inf)])  # Подсчитываем значения на поле
    s += np.sum(P[P > inf + th * 4])  # Дополнительная проверка

    return s,  # Возвращаем результат как кортеж

# Функция мутации для кораблей
def mutShips(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(0, 1) if (i+1) % 3 == 0 else random.randint(1, POLE_SIZE)
    return individual,

# Регистрация операций для генетического алгоритма
toolbox.register("evaluate", shipsFitness)  # Оценка
toolbox.register("select", tools.selTournament, tournsize=3)  # Селекция с турниром
toolbox.register("mate", tools.cxTwoPoint)  # Скрещивание (двухточечное)
toolbox.register("mutate", mutShips, indpb = 1.0 / LENGTH_CHROM)  # Мутация

# Статистика по популяции
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)  # Минимальная приспособленность
stats.register("avg", np.mean)  # Средняя приспособленность

# Функция для отображения кораблей на графике
def show(ax):
    ax.clear()
    show_ships(ax, hof.items[0], POLE_SIZE)  # Отображаем лучший индивид

    plt.draw()
    plt.gcf().canvas.flush_events()

# Включение интерактивного режима для отображения процесса
plt.ion()
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)

ax.set_xlim(-2, POLE_SIZE + 3)
ax.set_ylim(-2, POLE_SIZE + 3)

# Запуск генетического алгоритма с элитизмом
population, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                                 cxpb = P_CROSSOVER,
                                                 mutpb = P_MUTATION,
                                                 ngen = MAX_GENERATIONS,
                                                 halloffame = hof,
                                                 stats = stats,
                                                 callback = (show, (ax, )),  # Обновление графика после каждого поколения
                                                 verbose = True)

maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")

best = hof.items[0]
print(best)  # Вывод лучшего индивидуума

# Выключаем интерактивный режим и показываем график
plt.ioff()
plt.show()

# Если нужно отобразить график приспособленности, раскомментировать:
# plt.plot(maxFitnessValues, color = 'red')
# plt.plot(meanFitnessValues, color = 'green')
# plt.xlabel('Поколение')
# plt.ylabel('Макс/средняя приспособленность')
# plt.title('Зависимость максимальной и средней приспособленности от поколения')
# plt.show()
