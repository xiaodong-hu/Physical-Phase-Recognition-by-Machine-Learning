import os
import shutil
import multiprocessing as mp
import random

def make_data_dir():
    if os.path.exists('data'):
        char = input('directory /data already exits. Are you sure to cover it? (yes|no)\n')
        if char == 'yes':
            shutil.rmtree('data')
            os.mkdir('data')
    else:
        os.mkdir('data')

def run(i):
    T = str((float)(temperature_low) + (float)(i)*delta_T)
    output_order = str(i)
    os.system('./Ising'+' -n '+size+' -c '+cycle+' -t '+T+' -o '+output_order)
    j = len(os.listdir())
    if (j%50) == 0:
        print('{}\tdata has been generated!'.format(j))

def move_to_test_set(i):
    order = str(i);
    os.system('mv '+order+' '+'test\ set/'+order+'.csv')    # Note the space of file

def move_to_train_set(i):   
    order = str(i);
    os.system('mv '+order+' '+'training\ set/'+order+'.csv')# Note the space of file


if __name__ == "__main__":
    
    make_data_dir()
    os.system('cp Ising data/Ising')
    work_directory = os.getcwd()
    os.chdir(work_directory+'/data') # change directory to /data

    size = input('Input the size of lattice:(default is 10)\n')
    cycle = input('Input the time reaching equilibrium:(default is 2000)\n')
    temperature_low = input('Input the starting low temperature:')
    temperature_high = input('Input the final high temperature:')
    number = input('Input the number of configuration you want to generate:\n')
    propotion = input('Input the propotion of the generated date you want them belonging to the TRAINING SET:')
    while float(propotion)>1 or float(propotion)<0:
        propotion = input('Propotion of the Training Set must range from zero to one, Please reinput it:')
    print('\n\n')
    
    delta_T = ((float)(temperature_high)-(float)(temperature_low))/(float)(number)
    
    pool = mp.Pool(processes=8)
    pool.map(run,range(int(number)))                # Parallel generation
    
    # Random select generated files to obtain training set and test set
    training_set_list = random.sample(range(int(number)),int(int(number)*float(propotion)))
    # Use set to obtain the difference set of the list
    test_set_list = [i for i in set(range(int(number)))-set(training_set_list)]

    os.system('mkdir "test set"')
    os.system('mkdir "training set"')

    pool.map(move_to_train_set,training_set_list)   # Parallel Move files
    pool.map(move_to_test_set,test_set_list)        # Parallel Move files
    pool.close()
    pool.join()

os.system('rm Ising') # delete the temporary moved executable file
