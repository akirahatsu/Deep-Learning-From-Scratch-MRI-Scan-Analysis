import numpy as np

class Dense():
  
  # Initialization weight and bias
  def __init__(self , n_input , n_neuron):

    self.weight = np.random.randn(n_input ,n_neuron) * np.sqrt(2 / n_input) # randn == normal distribution 
    self.bias = np.zeros((n_neuron)) # 0 initially bias has no effect to Relu
  #Forwardpass
  def forward(self , input):   
    
    self.input = np.array( input)
    self.output = self.input @ self.weight + self.bias

    return self.output    
  #Backworpass
  def backworpass(self, dl_da ):
    
    #gradient with respect to weight --> dl/dw
    self.dl_dw = self.input.T @ dl_da # y=mx+b --> y'= x
    #gradient with respect to bias
    self.dl_db = np.sum(dl_da, axis=0, keepdims=True) # y=mx+b --> dy/db = 1 | dl/db = dl/da * 1 
    #gradient with respect to input
    self.dl_di = dl_da @ self.weight.T 
