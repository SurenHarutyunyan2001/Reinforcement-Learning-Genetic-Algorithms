# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')
import numpy as np

from deap import tools
from deap import algorithms

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats = None,
             halloffame = None, verbose = __debug__):
    """Этот алгоритм похож на алгоритм DEAP eaSimple(), с модификацией, что
    используется halloffame для реализации механизма элитизма. Индивиды,
    содержащиеся в halloffame, напрямую включаются в следующее поколение и не
    подвержены генетическим операторам выбора, кроссовера и мутации.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Оценка индивидуумов с недействительной приспособленностью
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("Параметр halloffame не должен быть пустым!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen = 0, nevals = len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Начало процесса поколений
    for gen in range(1, ngen + 1):

        # Выбор индивидуумов для следующего поколения
        offspring = toolbox.select(population, len(population) - hof_size)

        # Мутация и кроссовер в группе индивидов
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Оценка индивидуумов с недействительной приспособленностью
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Добавляем лучших обратно в популяцию:
        offspring.extend(halloffame.items)

        # Обновление halloffame с генерированными индивидуумами
        halloffame.update(offspring)

        # Замена текущей популяции на потомков
        population[:] = offspring

        # Добавление статистики текущего поколения в журнал
        record = stats.compile(population) if stats else {}
        logbook.record(gen = gen, nevals = len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
