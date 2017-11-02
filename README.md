# Physical-Phase-Recognition-by-Machine-Learning
Final Project of the class 数据科学导论


# Project Outline
[TOC]

## High/Low-temperature Phase Recognition by Logistic Regression

- Data (already in equilibrium), both for *training set* and *test set*, are all generated by **Monte Carlo simulation** (see *Notes of Computational Physics* by 丁泽军)
- Basic structure: 
    - Input layer $x_i$ is the configuration of $m\times m$ spins (flatten to be a one dimensional vector of $m^2$ components. Namely $\mathrm{dim}x_i=m^2$).
    - Output layer $y$ is a Boolean value determining wether one specific configuration is in the high or low temperature phase. In consideration of the dichotomy classification, we thus choose **Logistic Regression**: $$y=\mathrm{sigmoid}(Wx+b)$$ as our model.
- In light of the property of values of spins in Ising model, the input layer is recommed to be activated by $\mathrm{Relu}$ function.
- Output layer is certainly activated by $\mathrm{sigmoid}$ function.
- It is predicted that *shallow NN* should perform well enough. So perhaps we need no hidden layer. (This will dramatically simplifies our model and saves our time...)

## Topological Phase Recongition by CNN
