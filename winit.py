from math import sqrt
from numpy import random
random.seed(12345)


# ***************** CNN **********************
def lecun(w, h, ch, fan_in):
    m = 2.4/fan_in
    return random.uniform(low=-m, high=m, size=(w, h, ch)).tolist()
# ********************************************
def zeros(n):
    return [0] * n

def lecun_fc(n_in, n_out):
    return random.uniform(low=-sqrt(3/n_in), high=sqrt(3/n_in), size=(n_in, n_out)).tolist()

def he_uniform(n_in, n_out):
    limit = sqrt(6 / n_in) 
    return [[random.uniform(-limit, limit) for _ in range(n_out)] for _ in range(n_in)]

def he_normal(n_in, n_out):
    stddev = sqrt(2 / n_in)
    return [[random.gauss(0, stddev) for _ in range(n_out)] for _ in range(n_in)]

def bias_init(n_out, spread=1.0):
    return [random.uniform(-spread, spread) for _ in range(n_out)]
