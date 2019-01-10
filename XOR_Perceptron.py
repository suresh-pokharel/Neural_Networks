import numpy as np

# sigmoid function
# activation function
def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

# derivative of sigmoid function that
def sigmoid_der(x):
        return x*(1-x)

class NN:

    # default constructor
    # initialize weights
    def __init__(self,inputs={}):
        self.inputs = inputs
        # length of inputs,ie.4
        self.l = len(self.inputs)
        self.li = len(self.inputs[0])

        # initialize weights with random values
        self.w1 = np.random.random((self.li,self.l))
        print("W1="+str(self.w1))
        self.w2 = np.random.random((self.l, 1))
        print("W2=" + str(self.w2))

    # returns just sigmoid of inputs. ie. initial prediction
    def think(self):
        s1 = sigmoid(np.dot(self.inputs, self.w1 ))
        s2 = sigmoid(np.dot(s1, self.w2 ))
        return s2

    # training function
    def train(self, inputs, outputs, iterations):
        for i in range(iterations):
            l0 = inputs
            l1 = sigmoid(np.dot(l0, self.w1))
            l2 = sigmoid(np.dot(l1, self.w2))
            # find error
            l2_error = outputs - l2
            l2_delta = np.multiply(l2_error, sigmoid_der(l2))

            l1_error = np.multiply(l2_delta, self.w2.T)

            l1_delta = np.multiply(l1_error, sigmoid_der(l1))

            self.w2 = self.w2 + np.dot(l1.T, l2_delta)
            self.w1 = self.w1 + np.dot(l0.T, l1_delta)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([ [0], [1], [1], [0]])

obj = NN(inputs)
print("Before Training:\n")
print(obj.think())

obj.train(inputs,outputs, 1000000)

print("After Training:\n")
print(obj.think())


