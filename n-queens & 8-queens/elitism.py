# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

from deap import tools
from deap import algorithms

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats = None,
             halloffame = None, verbose = __debug__):
    # Этот алгоритм похож на алгоритм DEAP eaSimple(), с модификацией, что
    # используется механизм элитизма с halloffame. Индивиды, содержащиеся в halloffame,
    # напрямую вводятся в следующее поколение и не подвергаются генетическим операторам
    # выбора, кроссовера и мутации.
    
    logbook = tools.Logbook()  # Создание журнала для логирования
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])  # Заголовок для журнала

    # Оценка индивидуумов с недействительной приспособленностью
    invalid_ind = [ind for ind in population if not ind.fitness.valid]  # Список недействительных индивидуумов
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # Оценка фитнеса
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit  # Присвоение значений фитнеса

    if halloffame is None:
        raise ValueError("Параметр halloffame не должен быть пустым!")  # Проверка на наличие halloffame

    halloffame.update(population)  # Обновление halloffame с учетом текущей популяции
    hof_size = len(halloffame.items) if halloffame.items else 0  # Размер halloffame

    record = stats.compile(population) if stats else {}  # Сбор статистики для популяции
    logbook.record(gen=0, nevals=len(invalid_ind), **record)  # Запись статистики в журнал
    if verbose:
        print(logbook.stream)  # Вывод информации, если verbose включен

    # Начало процесса поколений
    for gen in range(1, ngen + 1):

        # Выбор индивидуумов для следующего поколения
        offspring = toolbox.select(population, len(population) - hof_size)  # Отбор потомков

        # Варьирование пула индивидуумов (кроссовер и мутация)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)  # Применение генетических операторов

        # Оценка индивидуумов с недействительной приспособленностью
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]  # Список недействительных индивидуумов
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # Оценка фитнеса
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit  # Присвоение значений фитнеса

        # Добавление лучших индивидуумов обратно в популяцию:
        offspring.extend(halloffame.items)  # Добавление элитных индивидуумов

        # Обновление halloffame с учетом новых индивидуумов
        halloffame.update(offspring)

        # Замена текущей популяции на потомков
        population[:] = offspring

        # Запись статистики текущего поколения в журнал
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)  # Запись в журнал
        if verbose:
            print(logbook.stream)  # Вывод информации, если verbose включен

    return population, logbook  # Возвращаем популяцию и журнал
