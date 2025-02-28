import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import gym
import numpy as np

# Создаем окружение FrozenLake
env = gym.make('FrozenLake-v1')

def value_iteration(env, gamma = 1.0):
    # Инициализируем таблицу значений нулями
    value_table = np.zeros(env.observation_space.n)
    no_of_iterations = 100000  # Максимальное количество итераций
    threshold = 1e-20  # Порог сходимости
    
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)  # Копируем текущую таблицу значений
        
        for state in range(env.observation_space.n):
            Q_value = []  # Список значений Q для каждого действия
            
            for action in range(env.action_space.n):
                next_states_rewards = []  # Храним возможные награды для следующего состояния
                
                for next_sr in env.P[state][action]:  # Проходим по всем возможным переходам
                    trans_prob, next_state, reward_prob, _ = next_sr
                    # Вычисляем ожидаемую ценность состояния
                    next_states_rewards.append(trans_prob * (reward_prob + gamma * updated_value_table[next_state]))
                
                Q_value.append(np.sum(next_states_rewards))  # Добавляем сумму наград в Q-значение
            
            value_table[state] = max(Q_value)  # Обновляем значение состояния максимальным Q-значением
        
        # Проверяем сходимость
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            print('Value-iteration сошелся на итерации # %d.' % (i + 1))
            break
    
    return value_table

def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)  # Инициализируем политику
    
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)  # Храним Q-значения для каждого действия
        
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                # Вычисляем Q-значение для действия
                Q_table[action] += trans_prob * (reward_prob + gamma * value_table[next_state])
        
        policy[state] = np.argmax(Q_table)  # Выбираем действие с максимальным Q-значением
    
    return policy

# Запускаем алгоритм итерации значений
optimal_value_function = value_iteration(env = env, gamma = 1.0)
# Извлекаем оптимальную политику
optimal_policy = extract_policy(optimal_value_function, gamma = 1.0)
# Выводим оптимальную стратегию
print(optimal_policy)
