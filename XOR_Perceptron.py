import numpy as np


# sigmoid function
# activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# derivative of sigmoid function that
def sigmoid_der(x):
    return x*(1-x)


def tanh(x):
    return (np.exp(-x) - np.exp(-x))/ (np.exp(-x) - np.exp(-x))


def tanh_der(x):
    return 1-tanh(x)*tanh(x)


class NN:
    # initialize weights
    def __init__(self):
        self.inputs = inputs

        self.inputs = 2
        self.hidden_neurons = 2
        self.output_neurons = 1
        self.learning_rate = 0.1

        # initialize weights between [-0.5,0.5], array size inputs*hidden_neurons
        self.hidden_layer_weights = np.random.uniform(-0.5, 0.5, (self.inputs, self.hidden_neurons))
        print('hidden layer weight')
        print(self.hidden_layer_weights)
        # [[0.25774477 - 0.16258581 - 0.03231841 - 0.37290148 - 0.1929745]
        # [-0.19097576  0.47019774  0.20510209  0.28022634  0.35644582]]

        self.output_layer_weights = np.random.uniform(-0.5, 0.5, (self.output_neurons, self.hidden_neurons))
        print('output layer weight')
        print(self.output_layer_weights)
        # [[0.17867141 - 0.02389593 - 0.34501892 - 0.34701407 - 0.06609539]]
        output_layer_error_grad = np.zeros(self.output_neurons)

    def train(self, inputs, outputs, epochs):
        for e in range(epochs):
            for input, output in zip(inputs, outputs):
                print('For input' + str(input))
                hidden_layer_output = np.zeros(self.hidden_neurons)
                output_layer_output = np.zeros(self.output_neurons)

                # forward pass from input to hidden layers
                for j in range(self.hidden_neurons):
                    sum = 0
                    for i in range(len(input)):
                        sum = sum + input[i] * self.hidden_layer_weights[i][j]
                    hidden_layer_output[j] = sum
                print('Hidden layer output')
                print(hidden_layer_output)

                # forward pass from hidden to output layer
                for k in range(self.output_neurons):
                    sum = 0
                    for j in range(len(hidden_layer_output)):
                        sum = sum + hidden_layer_output[j] * self.output_layer_weights[k][j]
                    output_layer_output[k] = sigmoid(sum)

                    # BACK PROPAGATION
                    # calculate error for the output
                    error = output - output_layer_output[k]

                    # calculate error gradient at op layer
                    error_gradient = sigmoid_der(output_layer_output[k]) * error

                    # weight adjustment for hidden to output layer
                    for m in range(self.hidden_neurons):
                        # change in error
                        delta_weight = self.learning_rate * error_gradient * output_layer_output

                        self.output_layer_weights[k][m] += delta_weight

                        # calculate error gradient for hidden layer neurons
                        sum = 0
                        for l in range (self.output_neurons):
                            sum = sum + error_gradient * self.output_layer_weights[l][m]

                        error_gradient_hidden_delta = sigmoid_der(hidden_layer_output[m]) * sum

                        for n in range(len(input)):
                            # error gradient at hidden layer
                            delta_weight_change = self.learning_rate * error_gradient_hidden_delta * input[n]
                            self.hidden_layer_weights[n][m] += delta_weight_change

        print("-------- After Training ----------")
        print('hidden layer weight')
        print(self.hidden_layer_weights)

        print('output layer weight')
        print(self.output_layer_weights)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([ [0], [1], [1], [0]])

obj = NN()
obj.train(inputs,outputs,1000)
