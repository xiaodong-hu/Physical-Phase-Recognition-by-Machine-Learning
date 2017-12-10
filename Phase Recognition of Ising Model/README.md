# Phase Recognition Module

## Usage:

---
First, you should know what **lattice size of data** you have generated, for example, file ```lattice size 20``` under ```Ising Model Data/data```, then
```shell
python Phase Recognition-Fully Connected.py -s 20
```
---
You are allowed to Recognize phases for different size and visualize them in ```tensorboard```, then you can continue typing (suppose ```lattice size 30``` file also exists)
```shell
python Phase Recognition-Fully Connected.py -s 25
```
in your terminal to generate tensorboard events files under ```/log```.

However, if you wanna clear all the tensorboard logs you have generated before, you can type in
```shell
python Phase Recognition-Fully Connected.py -d yes
```
to delete all the logs

---

As for training parameters such as batches and epoches, you can also manually set them by type in
```shell
python Phase Recognition-Fully Connected.py -s 25 -b 5 -e 200
```
---
To see the description of all the parameters, type in
```shell
python Phase Recognition-Fully Connected.py -h
```