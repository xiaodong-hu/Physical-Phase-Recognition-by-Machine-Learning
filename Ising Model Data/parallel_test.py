import multiprocessing as mp
import time
from math import sin,cos

def f(x):
    return sin(x)*cos(x)

cores = mp.cpu_count()              #number of processing on CPU
pool = mp.Pool(processes = cores)   #define parallel pools
n = input("input test range:\n")
test = range(int(n))
list = []
start = time.time()
for i in test:
    list.append(f(i))    
#print(list)
end = time.time()
print("串行时间",end-start)

start = time.time()
pool.map(f,test)           #single parameter use pool.map; multi-parameter use pool.apply
end = time.time()
print('并行时间',end-start)
pool.close()
pool.join()
