from layers import F_C, ReLU, Tanh, softmax, leaky_relu, leaky_relu_deriv
from types import SimpleNamespace


class FC():
    def __init__(self):
        self.l = []
        self.l.append(F_C(1, 12))
        self.l.append(ReLU())
        # self.l.append(Tanh())
        self.l.append(F_C(12, 1))
        
        

    
    def forward(self, input):
        out = input
        for l in self.l:
            out = l.forward(out)
        return out
    
    def forward4train(self, input):
        self.x = [input]
        for l in self.l:
            self.x.append(l.forward4train(self.x[-1]))

        return self.x[-1]
    
    def backward(self, grad_output):
        self.grad_x = [grad_output]
        for i, l in enumerate(self.l[::-1]):
            args = SimpleNamespace()
            args.grad_output = self.grad_x[-1]
            args.x = self.x[i]
            self.grad_x.append(l.backward(args))
        self.grad_x.reverse()

    def optimize(self, optimizer):
        for l in self.l:
            try:
                l.optimize(optimizer)
            except AttributeError:
                pass


