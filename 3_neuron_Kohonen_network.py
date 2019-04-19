# Competitive NN
# Three-neuron Kohonen network
# Self Organizing Map


import numpy as np


class Kohonen:
    def __init__(self, inputs, iterations, X):
        self.iterations = iterations
        self.learning_rate = 0.1
        self.inputs = inputs
        # initialize weights
        self.X = X

    def find_winner(self):
        for j in range(self.iterations):
            # initialize shortest distance to 0
            shortest_distance = 10000  # a large number
            delta = [0,0]  # to store update factor of weights

            for i in range(len(self.inputs)):
                distance = self.euclidean(self.inputs[i], self.X)
                print("Distance: " + str(distance))
                if distance < shortest_distance:
                    shortest_distance = distance
                    winner = i # stores which one is nearest

            # winner'th neuron is nearest so we need to update weights of Wi, ie. inputs[i]
            print("Shortest distance node: " +str(winner) + " --- distance: "+ str(shortest_distance))
            winning_node = self.inputs[winner]
            delta[0] = self.learning_rate * (self.X[0] - winning_node[0])  # delta_w_1_i
            delta[1] = self.learning_rate * (self.X[1] - winning_node[1])  # delta_w_2_i
            print("delta: " + str(delta))

            print("--------------------")
            # update weights of winner'th node
            self.inputs[winner] = np.add(self.inputs[winner], delta)

        return self.inputs

    def euclidean(self,x,y):
        return np.sqrt(np.square(y[0] - x[0]) + np.square(y[1] - x[1]))


inputs = [[.27, .81],[.42, .70], [.43, .21]]  # input weights
iterations = 9
X = [0.52, 0.12]

obj = Kohonen(inputs,iterations,X)
print(obj.find_winner())
