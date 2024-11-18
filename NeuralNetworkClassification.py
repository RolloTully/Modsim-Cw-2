import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class main():
    def __init__(self):
        '''initialisation'''
        self.layer_dims = np.array([3,20,1])
        #Layers
        self.Weights = [0.03*np.random((self.layer_dims[n+1],self.layer_dims[n])) - 0.015 for n in range(0, self.layer_dims.shape[0]-1)]
        self.Biases = [np.zeros((self.layer_dims[n],1)) for n in range(1,self.layer_dims.shape[0])]

        '''Test'''
        self.V_feature_vectors = []
        self.V_labels = []
        '''Train'''
        self.T_feature_vectors = []
        self.T_labels = []
    def train(self):

    def backward(self):

    def forward(self, x):
        self.a = [x]
        for n in range(0, self.Weights.shape[0]):
            self.a.append(self.grad_tanh(self.Weights[n]*self.a + self.Biases[n]))
        return self.a

    def cost(self,x):
        return (1/self.V_labels.shape[0])*np.sum((self.V_labels-self.forward(self.V_feature_vectors))**2)


    def grad_tanh(self, x):
        return 1 - np.tanh(x)**2
    def grad_sigmoid(self,x):
        pass
    def inv_grad_tanh(self, x):
        return

    def mainloop(self):


if __name__ == "__main__":
    main()
