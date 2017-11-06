import os
import shutil
import multiprocessing as mp

def makedir():
    if os.path.exists('data'):
        if input('directory already exits. Are you sure to cover it? (yes|no)\n')=='yes':
            shutil.rmtree('data')
            os.mkdir('data')
    else:
        os.mkdir('data')

def run(i):
    T = str((float)(temperature_low) + (float)(i)*delta_T)
    output_order = str(i)
    os.system('./Ising'+' -n '+size+' -c '+cycle+' -t '+T+' -o '+output_order)
    i = len(os.listdir())
    if (i%50) == 0:
        print('{}\tdata has been generated!'.format(i))

def move(i):
    order = str(i);
    os.system('mv '+order+' '+order+'.csv')


if __name__ == "__main__":
    
    makedir()
    os.system('cp Ising data/Ising')
    work_directory = os.getcwd()
    os.chdir(work_directory+'/data') # change directory to /data

    size = input('Input the size of lattice:(default is 10)\n')
    cycle = input('Input the time reaching equilibrium:(default is 2000)\n')
    temperature_low = input('Input the starting low temperature:')
    temperature_high = input('Input the final high temperature:')
    number = input('Input the number of configuration you want to generate:\n')
    print('\n\n')
    
    delta_T = ((float)(temperature_high)-(float)(temperature_low))/(float)(number)
    
    pool = mp.Pool(processes=8)
    pool.map(run,range(int(number)))
    pool.map(move,range(int(number)))
    pool.close()
    pool.join()

os.system('rm Ising') # delete the moved temporary executable file
