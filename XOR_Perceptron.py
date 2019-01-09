
# Multilayer Perceptron to implement XOR
# Ref:  https://github.com/Frixoe/xor-neural-network/blob/master/XOR-Net-Notebook.ipynb
import numpy as np


# Initilize weights
w11 = 1
w12 = 1
w21 = 1
w22 = 1

v1 = 1
v2 = 1

theta = 0.1

#input and output
x1 = [0,0,1,1]
x2 = [0,1,0,1]
z = [1, 0, 0, 1]

con=0

while con:
    zin1 = x1*w11+x2*w21
    zin2 = x1*w12+x2*w22
    for i in range(1,4):
        if(zin1 > theta):

def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))