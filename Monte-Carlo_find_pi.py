import random
 
def count_pi(n:int) -> float:
	# общее количество бросков
    i:int = 0
	# сколько из них попало в круг
    count:int = 0
    # пока мы не дошли до финального броска
    while i < n:
        # случайным образом получаем координаты x и y
        x:float = random.random()
        y:float = random.random()
        # проверяем, попали мы в круг или нет
        if (pow(x, 2) + pow(y, 2)) < 1:
			# если попали — увеличиваем счётчик на 1
            count += 1
		# в любом случае увеличиваем общий счётчик
        i += 1
    # считаем и возвращаем число пи
    return 4 * (count / n)
 

pi:int = count_pi(1000000)
print(pi)