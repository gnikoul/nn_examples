import winit
from math import exp

class F_C():
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.w = winit.lecun(n_in, n_out)
        self.b = winit.bias_init(n_out)
    
    def forward(self, input):
        return [sum([wi * iinput for wi, iinput in zip([row[l] for row in self.w], input)]) + self.b[l] for l in range(self.n_out)]

    def forward4train(self, input):
        self.last_input = input
        return self.forward(input)

    def backward(self, args):
        Lgrad_input = [0] * self.n_in
        self.dw = [[0 for _ in range(self.n_out)] for _ in range(self.n_in)]
        self.db = [0 for _ in range(self.n_out)]

        for i in range(self.n_in):
            for j in range(self.n_out):
                self.dw[i][j] = args.grad_output[j] * self.last_input[i] # dL/dwij
                Lgrad_input[i] += self.w[i][j] * args.grad_output[j]

        for j in range(self.n_out):
            self.db[j] = args.grad_output[j]

        return Lgrad_input

    def optimize(self, optimizer):
        optimizer.step(self.w, self.dw)
        optimizer.step(self.b, self.db)
        

class CNN():
    # https://colab.research.google.com/drive/1J53dKIqdllpbZ_XFizzeO5k6Ho1T7fhN
    def __init__(self, n_in, n_out):
        pass


# ReLU
class ReLU():
    def forward(self, x):
        return [max(0, xi) for xi in x]
    
    def forward4train(self, x):
        return self.forward(x)

    def backward(self, args):
        return [go if xi > 0 else 0 for xi, go in zip(args.x, args.grad_output)]

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return [xi if xi > 0 else alpha * xi for xi in x]

def leaky_relu_deriv(x, grad_output, alpha=0.01):
    return [go if xi > 0 else alpha * go for xi, go in zip(x, grad_output)]

import math

# Tanh
class Tanh():
    def forward(self, x):
        return [math.tanh(xi) for xi in x]
    
    def forward4train(self, x):
        return self.forward(x)

    def backward(self, args):
        return [go * (1 - math.tanh(xi) ** 2) for xi, go in zip(args.x, args.grad_output)]

# Softmax
def softmax(x):
    s = sum([exp(xi) for xi in x])
    return [exp(xi)/s for xi in x]