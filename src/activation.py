import numpy as np

#Relu activation 
class Relu():
  
  # Forward Relu 
  def forward(self ,input):
    self.input = input
    self.output = np.maximum(0,input) # f(x) = max(0,x)
    return self.output

  ''' 
  dl_da = gradient from next layer
  da_dz = ReLU derivative
  dl_dz = dl_da * da_dz
  '''

  #Backward Relu
  def backward(self,dl_da):

     self.da_dz = np.where(self.input > 0 , 1, 0)  # a = f(x) = max(0,x) --> df(x)/dx = 0 , x < 0 | 1 , x > 0 
     self.dl_dz = dl_da * self.da_dz # here we are finding loss with respect neuron | multiplicarion reson chain rule 
     return self.dl_dz

#Softmax 
class Softmax():
  
  #Forward 
  def forward(self, input):

    norm_input = input - np.max(input, axis = 1, keepdims=1 ) # shifing if input[n] huge  log will make overflow  input = [[1000, 999, 998]]
    
    exp = np.exp(norm_input) 
    
    self.output = exp / np.sum(exp , axis = 1 , keepdims = 1) #Normal dis --> raw sum == 1
    return self.output
