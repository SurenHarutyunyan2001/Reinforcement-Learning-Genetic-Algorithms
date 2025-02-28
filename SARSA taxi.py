# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import gym
import random
import numpy as np

env = gym.make('Taxi-v3')

# Гиперпараметры алгоритма SARSA
alpha = 0.85  # Скорость обучения
gamma = 0.90  # Коэффициент дисконтирования
epsilon = 0.8  # Вероятность случайного действия

# Инициализация Q-таблицы
Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s, a)] = 0.0

# Функция выбора действия по ε-жадной стратегии
def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(range(env.action_space.n), key = lambda x: Q[(state, x)])

# Обучение алгоритма SARSA
for i in range(4000):
    print(f"Шаги во время обучения: {i + 1}")
    r = 0
    state, _ = env.reset()  # reset() возвращает кортеж (state, info)
    action = epsilon_greedy(state, epsilon)  # Выбираем первое действие
    
    while True:
        nextstate, reward, terminated, truncated, _ = env.step(action)   
        done = bool(terminated or truncated)  # Преобразуем в стандартный тип bool
        nextaction = epsilon_greedy(nextstate, epsilon)  # Выбираем следующее действие
        
        # Записываем действия, сделанные агентом ДО обновления Q-таблицы
        if action == 0:
            print('Вверх')
        elif action == 1:
            print('Вправо')
        elif action == 2:
            print('Вниз')
        elif action == 3:
            print('Влево')
        elif action == 4:
            print('Взять пассажира')
        elif action == 5:
            print('Ожидание')
        
        # Обновление Q-таблицы по алгоритму SARSA
        Q[(state, action)] += alpha * (reward + gamma * Q[(nextstate, nextaction)] - Q[(state, action)])
        
        action = nextaction
        state = nextstate
        r += reward
        if done:
            break
    print("total reward:", r)  # Выводим общую награду за эпизод
env.close()




