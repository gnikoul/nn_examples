# Stochastic Gradient Descent
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, param, grad):
        for i in range(len(param)):
            if isinstance(param[i], list):  # matrix for w
                for j in range(len(param[i])):
                    param[i][j] -= self.lr * grad[i][j]
            else:  # vector for b
                param[i] -= self.lr * grad[i]
