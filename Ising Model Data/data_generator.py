import os
import shutil
import multiprocessing as mp

#if os.path.exists('/home/hxd/Documents/数据挖掘导论课题-Phase Recognition by Machine Learning/Ising Model Data/data'):
if os.path.exists('data'):
    if input('directory already exits. Are you sure to cover it? (yes|no)\n')=='yes':
        shutil.rmtree('data')
        os.mkdir('data')
else:
    os.mkdir('data')


os.system('cp Ising data/Ising')
work_directory = os.getcwd()
os.chdir(work_directory+'/data') # change directory to /data

size = input('Input the size of lattice:(default is 10)\n')
cycle = input('Input the time reaching equilibrium:(default is 2000)\n')
temperature_low = input('Input the starting low temperature:')
temperature_high = input('Input the final high temperature:')
number = input('Input the number of configuration you want to generate:\n')

delta_T = ((float)(temperature_high)-(float)(temperature_low))/(float)(number)

def run(i):
    T = str((float)(temperature_low) + (float)(i)*delta_T)
    output_order = str(i)
    os.system('./Ising'+' -n '+size+' -c '+cycle+' -t '+T+' -o '+output_order)
    #if (i+1)%50 == 0:
    #    print('{}\tdata has been generated!'.format(i+1))

def move(i):
    order = str(i);
    os.system('mv '+order+' '+'data/'+order+'.csv')


if __name__ == "__main__":
    pool = mp.Pool(processes=8)
    pool.map(run,range(int(number)))
    pool.close()
    pool.join()

os.system('rm Ising') # delete the moved temporary executable file
