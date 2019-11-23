import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt


class L1_Reg:

    def __init__(self,lr,iter,lmbd,deg):
        self.iter = iter
        self.lr=lr
        self.lmbd = lmbd
        self.deg = deg

    def calc_cost(self,X,Y,theta):
        predictions = np.dot(X,theta)
        cost = 0.5*np.sum((predictions-Y)**2)+self.lmbd*np.sum(np.absolute(theta))
        return cost

    def calc_gradient(self,X,Y,predictions,theta,index):
        gd = 0.0
        gd = np.dot(predictions-Y,X[:,index])
        gd = (gd+self.lmbd*np.sign(theta[index]))
        return gd

    def grad_desc(self,X,Y,theta):
        cost_hist = []
        m=len(Y)
        itr_hist = []
        prev_cost = 0
        for it in range(self.iter+1):
            predictions = np.dot(X,theta)
            for i in range(len(theta)):
                theta[i] = theta[i] - self.lr*self.calc_gradient(X,Y,predictions,theta,i)
            curr_cost = self.calc_cost(X,Y,theta)
            itr_hist.append(it)
            cost_hist.append(curr_cost)
            if it%50 == 0:
                print("Iteration :",it,"  Cost:",curr_cost)
            if abs(curr_cost - prev_cost) < 0.1:
                print("Final Cost:",curr_cost)
                break
            prev_cost = curr_cost
        return theta, itr_hist, cost_hist

    def get_coeff(self):
        X,Y,x_mean,y_mean,x_std,y_std,X_test,Y_test = preprocess("3D_spatial_network.csv",self.deg)
        theta = np.random.rand(len(X[0]))
        ntheta, ith, coh = self.grad_desc(X,Y,theta)
        print(ntheta)
        predictions = np.dot(X,ntheta)
        print("RMSE Train: ",sqrt(mean_squared_error(Y, predictions)))
        print("R2 Score Train: ",r2_score(Y,predictions))
        predictions = np.dot(X_test,ntheta)
        print("RMSE Test: ",sqrt(mean_squared_error(Y_test, predictions)))
        print("R2 Score Test: ",r2_score(Y_test,predictions))
        plt.plot(ith,coh)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Gradient Desc")
        plt.show()
        return ntheta,X_test,Y_test


if __name__ == "__main__":
    deg = int(input("Enter the degree:"))
    lr = float(input("Enter Learning rate:"))
    iter = int(input("Enter Max Iterations:"))
    lmd  =float(input("Enter regularization coeff:"))
    l1r = L1_Reg(lr,iter,lmd,deg)
    ntheta,X_test,Y_test = l1r.get_coeff()
