import winit
from winit import zeros
from math import exp
import math
import numpy as np

class FC():
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
                n_input_ch,
                n_feature_maps, 
                receptive_field_size: '[height, width]',
                step):
        self.n_feature_maps = n_feature_maps
        self.receptive_field_size = receptive_field_size
        self.step = step

        self.fan_in = self.receptive_field_size[0]*self.receptive_field_size[1]
        m = 2.4/self.fan_in
        size = (self.receptive_field_size[0], self.receptive_field_size[1], n_input_ch, self.n_feature_maps)

        self.kernel = winit.rand(low = -m, high= m, size=size)
        self.bias = winit.zeros(self.n_feature_maps)

    def forward(self, input: '[height][width][chanel]'):
        h_offset_front = int(self.receptive_field_size[0]/2)
        h_offset_back = int(self.receptive_field_size[0]/2) + (self.receptive_field_size[0]%2 - 1) 
        w_offset_front = int(self.receptive_field_size[1]/2)
        w_offset_back = int(self.receptive_field_size[1]/2) + (self.receptive_field_size[1]%2 - 1) 
        
        h_range = range(h_offset_back, len(input) - h_offset_front, self.step)
        w_range = range(w_offset_back, len(input[0]) - w_offset_front, self.step)
        fm_range = range(self.n_feature_maps)

        out = zeros((len(h_range), len(w_range), len(fm_range)))


        for fm in fm_range: # for each output fm
            for out_h, h in enumerate(h_range): # scan the image # out_h = h - h_offset_back - h*(self.step-1)
                for out_w, w in enumerate(w_range): # out_w = w - h_offset_back - w*(self.step-1)
                    for ch in range(len(input[0][0])): # for each input chanel
                        for i in range(-h_offset_back, h_offset_front + 1): # scan the receptive field
                            for j in range(-w_offset_back, w_offset_front + 1):
                                out[out_h][out_w][fm] += self.kernel[i + h_offset_back][j + w_offset_back][ch][fm] * input[h + i][w + j][ch]
                    out[out_h][out_w][fm] += self.bias[fm]

        return out
    
    def forward4train(self, input):
        self.last_input = input
        return self.forward(input)

    def backward(self, args):
        # args.grad_output shape: [out_h][out_w][n_feature_maps]
        input = self.last_input
        h_offset_front = int(self.receptive_field_size[0]/2)
        h_offset_back = int(self.receptive_field_size[0]/2) + (self.receptive_field_size[0]%2 - 1) 
        w_offset_front = int(self.receptive_field_size[1]/2)
        w_offset_back = int(self.receptive_field_size[1]/2) + (self.receptive_field_size[1]%2 - 1) 
        
        h_range = range(w_offset_back, len(input) - h_offset_front, self.step)
        w_range = range(w_offset_back, len(input[0]) - w_offset_front, self.step)
        fm_range = range(self.n_feature_maps)

        # Initialize gradients
        self.dkernel = np.zeros_like(self.kernel)
        self.dbias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(input)

        for fm in fm_range:
            for h_idx, h in enumerate(h_range):
                for w_idx, w in enumerate(w_range):
                    grad = args.grad_output[h_idx][w_idx][fm]
                    for ch in range(len(input[0][0])):
                        for i in range(-h_offset_back, h_offset_front + 1):
                            for j in range(-w_offset_back, w_offset_front + 1):
                                # Grad w.r.t. kernel
                                self.dkernel[i + h_offset_back][j + w_offset_back][ch][fm] += input[h + i][w + j][ch] * grad
                                # Grad w.r.t. input (for backprop)
                                grad_input[h + i][w + j][ch] += self.kernel[i + h_offset_back][j + w_offset_back][ch][fm] * grad
                    # Grad w.r.t. bias
                    self.dbias[fm] += grad

class AVRPOOL():
    def __init__(self,
                 n_feature_maps,
                 receptive_field_size: '[height, width]',
                 step):
        self.n_feature_maps = n_feature_maps
        self.receptive_field_size = receptive_field_size
        self.step = step

        self.fan_in = self.receptive_field_size[0]*self.receptive_field_size[1]
        self.coef = winit.rand(size=self.n_feature_maps)
        self.bias = winit.zeros(self.n_feature_maps)

    def forward(self, input: '[height][width][chanel]'):
        h_offset_front = int(self.receptive_field_size[0]/2)
        h_offset_back = int(self.receptive_field_size[0]/2) + (self.receptive_field_size[0]%2 - 1) 
        w_offset_front = int(self.receptive_field_size[1]/2)
        w_offset_back = int(self.receptive_field_size[1]/2) + (self.receptive_field_size[1]%2 - 1) 
        
        h_range = range(h_offset_back, len(input) - h_offset_front, self.step)
        w_range = range(w_offset_back, len(input[0]) - w_offset_front, self.step)
        fm_range = range(self.n_feature_maps)

        out = zeros((len(h_range), len(w_range), len(fm_range)))

        for fm in fm_range:
            for out_h, h in enumerate(h_range):
                for out_w, w in enumerate(w_range):
                    for ch in range(len(input[0][0])):
                        for i in range(-h_offset_back, h_offset_front + 1):
                            for j in range(-w_offset_back, w_offset_front + 1):
                                out[out_h][out_w][fm] += input[h + i][w + j][ch]
                    n_of_inp = self.receptive_field_size[0]*self.receptive_field_size[1]
                    out[out_h][out_w][fm] = self.coef[fm] * out[out_h][out_w][fm]/n_of_inp + self.bias[fm]   
        return out

    def forward4train(self, input):
        self.last_input = input
        return self.forward(input)

    def backward(self, args):
        # args.grad_output shape: [out_h][out_w][n_feature_maps]
        input = self.last_input
        h_offset_front = int(self.receptive_field_size[0]/2)
        h_offset_back = int(self.receptive_field_size[0]/2) + (self.receptive_field_size[0]%2 - 1) 
        w_offset_front = int(self.receptive_field_size[1]/2)
        w_offset_back = int(self.receptive_field_size[1]/2) + (self.receptive_field_size[1]%2 - 1) 
        
        h_range = range(w_offset_back, len(input) - h_offset_front, self.step)
        w_range = range(w_offset_back, len(input[0]) - w_offset_front, self.step)
        fm_range = range(self.n_feature_maps)

        n_of_inp = self.receptive_field_size[0]*self.receptive_field_size[1]
        grad_input = zeros_like(input)
        self.dcoef = zeros_like(self.coef)
        self.dbias = zeros_like(self.bias)

        for fm in fm_range:
            for out_h, h in enumerate(h_range):
                for out_w, w in enumerate(w_range):
                    grad = args.grad_output[out_h][out_w][fm]
                    # Gradient w.r.t. coef
                    sum_pool = 0
                    for ch in range(len(input[0][0])):
                        for i in range(-h_offset_back, h_offset_front + 1):
                            for j in range(-w_offset_back, w_offset_front + 1):
                                sum_pool += input[h + i][w + j][ch]
                                # Gradient w.r.t. input
                                grad_input[h + i][w + j][ch] += grad * self.coef[fm] / n_of_inp
                    self.dcoef[fm] += grad * sum_pool / n_of_inp
                    # Gradient w.r.t. bias
                    self.dbias[fm] += grad

        return grad_input

    def optimize(self, optimizer):
        optimizer.step(self.coef, self.dcoef)
        optimizer.step(self.bias, self.dbias)

# Activation
def apply_recursive(x, func):
    if isinstance(x, list):
        return [apply_recursive(xi, func) for xi in x]
    else:
        return func(x)

def zip_recursive(x, y, func):
    if isinstance(x, list) and isinstance(y, list):
        return [zip_recursive(xi, yi, func) for xi, yi in zip(x, y)]
    else:
        return func(x, y)

class ReLU():
    def forward(self, x):
        return apply_recursive(x, lambda xi: max(0, xi))
    def forward4train(self, x):
        return self.forward(x)
    def backward(self, args):
        return zip_recursive(args.x, args.grad_output, lambda xi, go: go if xi > 0 else 0)



class Leaky_ReLU():
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def forward(self, x):
        return apply_recursive(x, lambda xi: xi if xi > 0 else self.alpha * xi)
    def backward(self, args):
        return zip_recursive(args.x, args.grad_output,
                             lambda xi, go: go if xi > 0 else self.alpha * go)



class Tanh():
    def __init__(self, a = 1, s = 1):
        self.a = a
        self.s = s
    def forward(self, x):
        return apply_recursive(x, lambda xi: self.a * math.tanh(self.s * xi))
    def forward4train(self, x):
        return self.forward(x)
    def backward(self, args):
        return zip_recursive(
            args.x, args.grad_output,
            lambda xi, go: go * self.a * self.s * (1 - math.tanh(self.s * xi) ** 2))


class Sigmoid():
    def forward(self, x):
        return apply_recursive(x, lambda xi: 1 / (1 + math.exp(-xi)))
    def forward4train(self, x):
        return self.forward(x)
    def backward(self, args):
        return zip_recursive(
            args.x, args.grad_output,
            lambda xi, go: go * self.sigmoid_scalar(xi) * (1 - self.sigmoid_scalar(xi)))
    def sigmoid_scalar(self, xi):
        return 1 / (1 + math.exp(-xi))


# Classification
def softmax(x):
    s = sum([exp(xi) for xi in x])
    return [exp(xi)/s for xi in x]


# General 

def shape(x):
    shape = []
    while isinstance(x, list):
        shape.append(len(x))
        if len(x) == 0:
            break
        x = x[0]
    return tuple(shape)

def flatten(x):
    if isinstance(x, list):
        result = []
        for xi in x:
            result.extend(flatten(xi))
        return result
    else:
        return [x]
