# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from deap import tools
from deap.algorithms import varAnd


def eaSimpleElitism(population, toolbox, cxpb, mutpb, ngen, stats = None,
             halloffame = None, verbose = __debug__, callback = None):
    # Переработанный алгоритм eaSimple с элементом элитизма
    

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Оценка индивидов с недействительной фитнес-функцией
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen = 0, nevals = len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Начало процесса генерации
    for gen in range(1, ngen + 1):
        # Выбор индивидов для следующего поколения
        offspring = toolbox.select(population, len(population) - hof_size)

        # Применение операторов кроссовера и мутации
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Оценка индивидов с недействительной фитнес-функцией
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)

        # Обновление halloffame с новыми индивидами
        if halloffame is not None:
            halloffame.update(offspring)

        # Замена текущей популяции на потомков
        population[:] = offspring

        # Добавление статистики текущего поколения в журнал
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if callback:
            callback[0](*callback[1])

    return population, logbook
