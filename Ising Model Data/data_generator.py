import os
import shutil
import multiprocessing as mp
import random

def new_mkdir(filename):
    if os.path.exists(filename):
        char = input('directory '+filename+' already exits. Are you sure to cover it? (yes|no)\n')
        if char == 'yes':
            shutil.rmtree(filename)
            os.mkdir(filename)
        elif char == 'no':
            if len(os.listdir()) > 0:  # check if sub_directory is empty
                map(shutil.rmtree, os.listdir()) # use map to delete groups of files
    else:
        os.mkdir(filename)

def run(i, size):
    delta_T = ((float)(temperature_high)-(float)(temperature_low))/(float)(number)
    T = str((float)(temperature_low) + (float)(i)*delta_T)
    output_order = str(i)
    os.system('./Ising'+' -n '+size+' -c '+cycle+' -t '+T+' -o '+output_order)
    j = len(os.listdir())
    if (j%50) == 0:
        print('{}\tdata has been generated!'.format(j))

def parallel_generate(size):

    pool = mp.Pool(processes=8)
    # prepare multi-argument mapping
    # cf. https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments for details
    parameter_list = [(i, size) for i in range(int(number))]
    pool.starmap(run, parameter_list)                                       # Parallel generation
    

def move_to_test_set(i, path):
    order = str(i);
    # Sample  mv 0 lattice\ size\ 20/test\ set/0.csv
    os.system('mv '+order+' '+path+'/test\ set/'+order+'.csv')      # Note the space of file

def move_to_train_set(i, path):
    order = str(i);
    os.system('mv '+order+' '+path+'/training\ set/'+order+'.csv')  # Note the space of file

def parallel_move(data_file_name):
    # Random select generated files to obtain training set and test set
    training_set_list = random.sample(range(int(number)),int(int(number)*float(propotion)))
    # Use set to obtain the difference set of the list
    test_set_list = [i for i in set(range(int(number)))-set(training_set_list)]
 
    os.system('mkdir '+'"lattice size '+data_file_name+'"')
    work_directory = os.getcwd()
    os.chdir(work_directory+'/lattice size '+data_file_name)     # change directory to /data/data_file_name
    os.system('mkdir "test set"')
    os.system('mkdir "training set"')
    os.chdir(os.path.pardir)                        # go back to data

    path = 'lattice\ size\ '+data_file_name
    pool = mp.Pool(processes=8)
    # prepare multi-argument mapping
    parameter_list = [(i, path) for i in training_set_list]
    pool.starmap(move_to_train_set, parameter_list) # Parallel Move files

    parameter_list = [(i, path) for i in test_set_list]
    pool.starmap(move_to_test_set, parameter_list)  # Parallel Move files
    pool.close()
    pool.join()


if __name__ == "__main__":
    
    new_mkdir('data')

    size_down = input('Input the Minimum size of lattice:(default is 10)\n')
    size_up = input('Input the Maximum size of lattice:(default is 20)\n')
    size_step = input('Input the step size:(default is 5)\n')
    cycle = input('Input the time reaching equilibrium:(default is 2000)\n')
    temperature_low = input('Input the starting low temperature:')
    temperature_high = input('Input the final high temperature:')
    number = input('Input the number of configuration you want to generate:\n')
    propotion = input('Input the propotion of the generated date you want them belonging to the TRAINING SET:')
    while float(propotion)>1 or float(propotion)<0:
        propotion = input('Propotion of the Training Set must range from zero to one, Please reinput it:')
    print('\n\n')

    os.system('cp Ising data/Ising')                # Temporarily copy file Ising to /data/Ising
    work_directory = os.getcwd()
    os.chdir(work_directory+'/data')                # change directory to /data

    for size in range(int(size_down),int(size_up),int(size_step)):
        print('Generating Data for Lattice Size {}'.format(size))
        parallel_generate(str(size))
        parallel_move(str(size))

    # still work in /data
    os.system('rm Ising') # delete the temporary moved executable file



