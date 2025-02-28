# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from deap import tools
from deap import algorithms

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats = None,
             halloffame = None, verbose = __debug__):
    # Этот алгоритм похож на стандартный алгоритм eaSimple() из DEAP, с изменением, что
    # используется механизм элитизма через halloffame. Индивиды, содержащиеся в
    # halloffame, непосредственно встраиваются в следующее поколение и не подвергаются
    # генетическим операторам выбора, кроссовера и мутации.
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Оценка индивидов с недействительной фитнес-функцией
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("Параметр halloffame не может быть пустым!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen = 0, nevals = len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Начинаем процесс генерации
    for gen in range(1, ngen + 1):

        # Выбираем индивидов для следующего поколения
        offspring = toolbox.select(population, len(population) - hof_size)

        # Применяем операторы мутации и кроссовера
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Оценка индивидов с недействительной фитнес-функцией
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Добавляем лучшие индивиды обратно в популяцию:
        offspring.extend(halloffame.items)

        # Обновляем halloffame с новыми индивидами
        halloffame.update(offspring)

        # Заменяем текущую популяцию на потомков
        population[:] = offspring

        # Добавляем статистику текущего поколения в журнал
        record = stats.compile(population) if stats else {}
        logbook.record(gen = gen, nevals = len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
