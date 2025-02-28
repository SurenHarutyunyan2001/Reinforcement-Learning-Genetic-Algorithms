# Установите старую версию numpy (например, 1.21.0), где np.bool8 поддерживается:
# pip install numpy==1.21.0

import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import random
import gym

# Создаём среду Taxi-v3 в текстовом режиме рендеринга (ansi)
env = gym.make("Taxi-v3", render_mode = "ansi")

# Инициализируем среду (reset обязателен перед render)
state, _ = env.reset()
print(env.render())  # Выводим текущее состояние среды в текстовом виде

# Параметры обучения Q-learning
alpha = 0.4  # Коэффициент обучения
gamma = 0.999  # Коэффициент дисконтирования (будущая награда)
epsilon = 0.017  # Вероятность случайного выбора действия (ε-жадная стратегия)

# Создаём Q-таблицу и заполняем её нулями
q = {}
for s in range(env.observation_space.n):  # Перебираем все возможные состояния
    for a in range(env.action_space.n):  # Перебираем все возможные действия
        q[(s, a)] = 0.0  # Начальное значение Q(s, a) = 0

# Функция обновления Q-таблицы по формуле Q-learning
def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
    # Выбираем наилучшую оценку Q для следующего состояния
    qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
    # Обновляем значение Q для текущего состояния и действия
    q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])

# Функция выбора действия по ε-жадной стратегии
def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:  # Случайное действие с вероятностью ε
        return env.action_space.sample()
    else:  # Иначе выбираем действие с максимальным значением Q(s, a)
        return max(range(env.action_space.n), key = lambda x: q[(state, x)])

# Запуск обучения на 8000 эпизодов
for i in range(8000):
    print(f"Шаги во время обучения: {i + 1}")
    r = 0  # Общая награда за эпизод
    prev_state, _ = env.reset()  # Начальное состояние
    while True:
        action = epsilon_greedy_policy(prev_state, epsilon)  # Выбираем действие
        nextstate, reward, done, _, _ = env.step(action)  # Делаем шаг в среде
        # Записываем действия, сделанные агентом
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
        update_q_table(prev_state, action, reward, nextstate, alpha, gamma)  # Обновляем Q-таблицу
        prev_state = nextstate  # Переход в следующее состояние
        r += reward  # Суммируем награду
        if done:  # Если эпизод завершён, выходим из цикла
            break
    print("total reward:", r)  # Выводим общую награду за эпизод

env.close()  # Закрываем среду после обучения
