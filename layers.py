import winit
from math import exp

class F_C():
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.w = winit.lecun_fc(n_in, n_out)
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
    # https://ieeexplore.ieee.org/document/726791
    def __init__(self, 
                input_size: '[height, width, chanel]', 
                n_feature_maps, 
                receptive_field_size: '[height, width]',
                step):
        self.input_size = input_size
        self.n_feature_maps = n_feature_maps
        self.receptive_field_size = receptive_field_size
        self.step = step

        self.fan_in = self.input_size[0]*self.input_size[1]*self.input_size[2]
        self.kernel = winit.lecun(self.input_size[0], self.input_size[1], self.n_feature_maps, self.fan_in)
        self.bias = [0]*self.n_feature_maps

    def forward(self, input: '[height][width][chanel]'):
        h_offset = int(self.receptive_field_size[0]/2)
        w_offset = int(self.receptive_field_size[1]/2)
        out = [[[0 for _ in range(self.n_feature_maps)] 
            for _ in range(self.input_size[1] - 2*w_offset)] 
            for _ in range(self.input_size[0] - 2*h_offset)]


        for fm in range(self.n_feature_maps): # for each output fm
            for h in range(self.input_size[0] - 2*h_offset): # scan the image
                for w in range(self.input_size[1] - 2*w_offset):
                    for ch in range(self.input_size[2]): # for each input chanel
                        for i in range(-h_offset, h_offset + 1): # scan the receptive field
                            for j in range(-w_offset, w_offset + 1):
                                out[h][w][fm] += self.kernel[h][w][fm] * input[h + i][w + j][ch]
                out[h][w][fm] += self.bias[fm]

        return out


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