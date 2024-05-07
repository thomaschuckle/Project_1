import numpy as np
import copy

class LinearModel:
    def __init__(self) -> None:
        pass
    
    def zscore_normalize(self, x):
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        x_normalized = (x-mu)/sigma
        return x_normalized, mu, sigma

    def compute_fwb(self, x_train, w, b):
        m = x_train.shape[0]
        fwb = np.zeros(m)
        for i in range(m):
            fwb[i] = np.dot(w, x_train[i]) + b
        return fwb
    
    def compute_cost(self, x_train, y_train, w, b):
        fwb = self.compute_fwb(x_train, w, b)
        m = x_train.shape[0]
        error = 0
        for i in range(m):
            error += (fwb[i] - y_train[i])**2
        cost = error/(2*m)
        return cost
    
    def compute_gradient(self, x_train, y_train, w, b):
        m, n = x_train.shape
        dj_dw = np.zeros(n)
        dj_db = 0
        fwb = self.compute_fwb(x_train, w, b)
        for i in range(m):
            error = fwb[i] - y_train[i]
            for j in range(n):
                dj_dw[j] += error*x_train[i][j]
            dj_db += error
        dj_dw = dj_dw/m
        dj_db = dj_db/m
        return dj_dw, dj_db

    def gradient_descent(self, x_train, y_train, w_ini, b_ini, alpha, iterations):
        w = copy.deepcopy(w_ini)
        b = b_ini
        cost = np.zeros(iterations)
        for i in range(iterations):
            dj_dw, dj_db = self.compute_gradient(x_train, y_train, w, b)
            w = w - alpha*dj_dw
            b = b - alpha*dj_db
            
            cost[i] = self.compute_cost(x_train, y_train, w, b)
        
        return w, b, cost
    