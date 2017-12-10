# Ising Model Data
Data generation by Monte Carlo costs long time, so this module is written by ```C``` and **parallel scheduled** by ```python```.

## Usage
If you wanna generate only one configuration for test, type in
```shell
./Ising -n (size) -c (cycles to reach equilibrium) -t (temperature) -o (output name)
```
If you wanna generate large number of files, type in
```shell
python data_generator.py
```
and input the parameters.

---

Data is stored in the file named by lattice sizes as ```lattice size 20``` for example and is automatically and **randomly** divided by two parts: **training set**, and **test set**.