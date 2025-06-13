from fc import FC
from lossfunctions import mse, dmse
from optimizers import SGD
import numpy as np

nn = FC()
# x = 1*y + 0
data = [[-3], [-2],[-1], [1],[2], [3], [4],[5]]  
targs = [[5], [-1],[-0.5], [2],[4], [30], [10],[20]] 
optimizer = SGD(lr = 0.0001)


for i in range(100000):
    total_loss = 0
    preds = []

    for x, t in zip(data, targs):
        # Forward pass
        pred = nn.forward4train(x)
        preds.append(pred[0])

        # Compute loss
        total_loss += mse([t[0]], [pred[0]])

        # Backward pass
        grad = dmse([t[0]], [pred[0]]) 
        nn.backward(grad)

        # Optimizer step
        nn.optimize(optimizer)

    print(i, ": ", total_loss)
print(preds)
        


import matplotlib.pyplot as plt

def plot_model_outputs(model, x_values):
    y_values = [model.forward([x])[0] for x in x_values]

    plt.plot(x_values, y_values, marker='', label='Model Output')
    flat_data = [d[0] for d in data]
    flat_targs = [t[0] for t in targs]
    plt.scatter(flat_data, flat_targs, color='red', label='Target Points', marker='*', s=200)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model Output vs Input")
    plt.grid(True)
    plt.legend()
    plt.show()


x_vals = np.arange(-6, 6, 0.1).tolist()
plot_model_outputs(nn, x_vals)