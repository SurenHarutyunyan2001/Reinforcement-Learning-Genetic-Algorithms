# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')
import numpy as np

class NurseSchedulingProblem:
    # Этот класс инкапсулирует задачу планирования смен медсестёр

    def __init__(self, hardConstraintPenalty):
        # param hardConstraintPenalty: коэффициент штрафа за нарушение жёсткого ограничения
        
        self.hardConstraintPenalty = hardConstraintPenalty

        # список медсестёр:
        self.nurses = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        # предпочтения медсестёр по сменам - утренним, вечерним, ночным:
        self.shiftPreference = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]]

        # минимальное и максимальное количество медсестёр на каждую смену - утреннюю, вечернюю, ночную:
        self.shiftMin = [2, 2, 1]
        self.shiftMax = [3, 4, 2]

        # максимальное количество смен в неделю для каждой медсестры
        self.maxShiftsPerWeek = 5

        # количество недель, для которых создаётся расписание:
        self.weeks = 1

        # полезные значения:
        self.shiftPerDay = len(self.shiftMin)
        self.shiftsPerWeek = 7 * self.shiftPerDay

    def __len__(self):
        # return: количество смен в расписании
        return len(self.nurses) * self.shiftsPerWeek * self.weeks


    def getCost(self, schedule):
        # Рассчитывает общую стоимость различных нарушений в данном расписании
        # param schedule: список бинарных значений, описывающих данное расписание
        # return: рассчитанная стоимость

        if len(schedule) != self.__len__():
            raise ValueError("Размер списка расписания должен быть равен ", self.__len__())

        # преобразуем всё расписание в словарь с отдельным расписанием для каждой медсестры:
        nurseShiftsDict = self.getNurseShifts(schedule)

        # считаем различные нарушения:
        consecutiveShiftViolations = self.countConsecutiveShiftViolations(nurseShiftsDict)
        shiftsPerWeekViolations = self.countShiftsPerWeekViolations(nurseShiftsDict)[1]
        nursesPerShiftViolations = self.countNursesPerShiftViolations(nurseShiftsDict)[1]
        shiftPreferenceViolations = self.countShiftPreferenceViolations(nurseShiftsDict)

        # рассчитываем стоимость нарушений:
        hardContstraintViolations = consecutiveShiftViolations + nursesPerShiftViolations + shiftsPerWeekViolations
        softContstraintViolations = shiftPreferenceViolations

        return self.hardConstraintPenalty * hardContstraintViolations + softContstraintViolations

    def getNurseShifts(self, schedule):
        # Преобразует всё расписание в словарь с отдельным расписанием для каждой медсестры
        # param schedule: список бинарных значений, описывающих данное расписание
        # return: словарь, где каждая медсестра является ключом, а соответствующие смены - значением
        
        shiftsPerNurse = self.__len__() // len(self.nurses)
        nurseShiftsDict = {}
        shiftIndex = 0

        for nurse in self.nurses:
            nurseShiftsDict[nurse] = schedule[shiftIndex:shiftIndex + shiftsPerNurse]
            shiftIndex += shiftsPerNurse

        return nurseShiftsDict

    def countConsecutiveShiftViolations(self, nurseShiftsDict):
        # Считает нарушения по последовательности смен в расписании
        # param nurseShiftsDict: словарь с отдельным расписанием для каждой медсестры
        # return: количество найденных нарушений
        
        violations = 0
        # проходим по сменам каждой медсестры:
        for nurseShifts in nurseShiftsDict.values():
            # ищем два последовательных '1':
            for shift1, shift2 in zip(nurseShifts, nurseShifts[1:]):
                if shift1 == 1 and shift2 == 1:
                    violations += 1
        return violations

    def countShiftsPerWeekViolations(self, nurseShiftsDict):
        # Считает нарушения по количеству смен в неделю для каждой медсестры
        # param nurseShiftsDict: словарь с отдельным расписанием для каждой медсестры
        # return: количество найденных нарушений
        
        violations = 0
        weeklyShiftsList = []
        # проходим по сменам каждой медсестры:
        for nurseShifts in nurseShiftsDict.values():  # все смены одной медсестры
            # проходим по сменам каждой недели:
            for i in range(0, self.weeks * self.shiftsPerWeek, self.shiftsPerWeek):
                # считаем все '1' за неделю:
                weeklyShifts = sum(nurseShifts[i : i + self.shiftsPerWeek])
                weeklyShiftsList.append(weeklyShifts)
                if weeklyShifts > self.maxShiftsPerWeek:
                    violations += weeklyShifts - self.maxShiftsPerWeek

        return weeklyShiftsList, violations

    def countNursesPerShiftViolations(self, nurseShiftsDict):
        # Считает нарушения по количеству медсестёр на каждую смену
        # param nurseShiftsDict: словарь с отдельным расписанием для каждой медсестры
        # return: количество найденных нарушений
        
        # суммируем смены всех медсестёр:
        totalPerShiftList = [sum(shift) for shift in zip(*nurseShiftsDict.values())]

        violations = 0
        # проходим по всем сменам и считаем нарушения:
        for shiftIndex, numOfNurses in enumerate(totalPerShiftList):
            dailyShiftIndex = shiftIndex % self.shiftPerDay  # -> 0, 1 или 2 для 3 смен в день
            if (numOfNurses > self.shiftMax[dailyShiftIndex]):
                violations += numOfNurses - self.shiftMax[dailyShiftIndex]
            elif (numOfNurses < self.shiftMin[dailyShiftIndex]):
                violations += self.shiftMin[dailyShiftIndex] - numOfNurses

        return totalPerShiftList, violations

    def countShiftPreferenceViolations(self, nurseShiftsDict):
        #Считает нарушения предпочтений медсестёр в расписании
        # param nurseShiftsDict: словарь с отдельным расписанием для каждой медсестры
        # return: количество найденных нарушений
        
        violations = 0
        for nurseIndex, shiftPreference in enumerate(self.shiftPreference):
            # дублируем предпочтения по сменам на дни периода
            preference = shiftPreference * (self.shiftsPerWeek // self.shiftPerDay)
            # проходим по сменам и сравниваем с предпочтениями:
            shifts = nurseShiftsDict[self.nurses[nurseIndex]]
            for pref, shift in zip(preference, shifts):
                if pref == 0 and shift == 1:
                    violations += 1

        return violations

    def printScheduleInfo(self, schedule):
        # Выводит информацию о расписании и нарушениях
        # param schedule: список бинарных значений, описывающих данное расписание
        
        nurseShiftsDict = self.getNurseShifts(schedule)

        print("Расписание для каждой медсестры:")
        for nurse in nurseShiftsDict:  # все смены одной медсестры
            print(nurse, ":", nurseShiftsDict[nurse])

        print("Нарушения по последовательности смен = ", self.countConsecutiveShiftViolations(nurseShiftsDict))
        print()

        weeklyShiftsList, violations = self.countShiftsPerWeekViolations(nurseShiftsDict)
        print("Смены по неделям = ", weeklyShiftsList)
        print("Нарушения по сменам в неделю = ", violations)
        print()

        totalPerShiftList, violations = self.countNursesPerShiftViolations(nurseShiftsDict)
        print("Медсестры на смену = ", totalPerShiftList)
        print("Нарушения по количеству медсестёр на смену = ", violations)
        print()

        shiftPreferenceViolations = self.countShiftPreferenceViolations(nurseShiftsDict)
        print("Нарушения предпочтений смен = ", shiftPreferenceViolations)
        print()


# тестирование класса:
def main():
    # создаём экземпляр задачи:
    nurses = NurseSchedulingProblem(10)

    randomSolution = np.random.randint(2, size=len(nurses))
    print("Случайное решение = ")
    print(randomSolution)
    print()

    nurses.printScheduleInfo(randomSolution)

    print("Общая стоимость = ", nurses.getCost(randomSolution))


if __name__ == "__main__":
    main()
