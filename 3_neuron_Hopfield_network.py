# Three-neuron Hopfield network
# memory upto 3 stages

import numpy as np


class Hopfield:
    def __init__(self, y):
        self.y = y
        self.memory = 3 # number of states to be memorised by the network
        self.I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def calculate(self):
        sum = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for i in range(self.memory):
            y = np.array(self.y[i]).reshape(1,3)  # convert 1D to 1*3
            transpose = np.array(y).T.reshape(3,1)  # transpose of 1*3 reshaping into 3*1
            sum = np.add(sum, np.multiply(transpose,y))

        #  calculate memory*I (multiply all elements by M)
        arr = self.I
        for i in range(len(arr)):
            arr[i] = [self.memory * x for x in arr[i]]

        result = np.subtract(sum,arr)

        print(result)


y = [[1, 1, 1],[-1, -1, -1], [1, -1, -1]]  # input for network [y1,y2]


obj = Hopfield(y)
obj.calculate()
