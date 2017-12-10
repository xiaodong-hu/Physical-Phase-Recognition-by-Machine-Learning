import multiprocessing as mp


def run(a, name):
	if a%2 == 0:
		print(name)
	else:
		print('0')

if __name__ == '__main__':
	number = 1000
	pool = mp.Pool(processes=8)
	size = 'test'
	parameter_list = [(i, size) for i in range(int(number))]
	pool.starmap(run, parameter_list)