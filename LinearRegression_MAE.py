import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class main():
    def __init__(self):
        '''Initialise parameters'''
        self.epochs = 10000
        self.alpha = 0.01
        self.params = np.array([0,0])
        self.w = 0
        self.b = 0
        '''Load data'''
        self.df = pd.read_csv('window_heat.csv') #reads in csv
        self.dt = self.df['dT[C]'].to_numpy() #converts to numpy arrays
        self.qdot = self.df['Qdot[W]'].to_numpy()
        self.train()
    def train(self):
        '''Uses SGD to update w and b values'''
        for _ in range(self.epochs):
            #calculates gradients
            print("Epoch: ",_+1," W:",self.w," B: ",self.b)
            self.n = int(np.random.rand()*self.df.shape[0])#selecting a random sample
            self.w = self.w - self.alpha*-np.sign(self.qdot[self.n]-(self.w*self.dt[self.n]+self.b))*self.dt[self.n]
            self.b = self.b - self.alpha*-np.sign(self.qdot[self.n]-(self.w*self.dt[self.n]+self.b))
        self.x = np.linspace(np.min(self.dt),np.max(self.dt),5)
        plt.plot(self.x, self.w*self.x+self.b)
        plt.scatter(self.dt,self.qdot)
        plt.show()

if __name__ == "__main__":
    main()
