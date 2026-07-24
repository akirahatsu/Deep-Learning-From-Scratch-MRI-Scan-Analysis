class CNN():

  def __init__(self, kernel_shape):
  
    self.kernel = np.random.randn(*kernel_shape)
    self.bias = np.zeros((kernel_shape[0]))

  def forward(self,input):
    
    self.signal_shape = np.array(input).shape 
    
    self.output = np.zeros(
       (self.kernel.shape[0] ,
       self.signal_shape[0]-self.kernel.shape[1] + 1,
      self.signal_shape[1]-self.kernel.shape[2] + 1))

    for k in range(self.kernel.shape[0]):
   
  

        for el in range(self.signal_shape[0]-self.kernel.shape[1] + 1):
            for i in range(self.signal_shape[1]-self.kernel.shape[2] + 1):
      
                patch = (input[
                     el : el+self.kernel.shape[1] ,
                     i :i+self.kernel.shape[2]
                     ])
    
                self.output[k ,el,i] = np.sum(patch * self.kernel[k]) + self.bias[k]
    return self.output

    
