class CatergoricalCrossEntropy():

  #Forward
  def forward(self,inputs,target):
    
    self.input = inputs
    target = np.array(target)
    
    #if dim == 2 means one hot encoded 
    if target.ndim == 2:

      encode = np.sum(target * inputs , axis = 1 , keepdims = 1) 

    elif target.ndim == 1:
      #reshape to broudcast
      encode = inputs[
          range(len(target)) ,target
          ].reshape(-1,1)  
       
    encode = np.clip(encode , 1e-7 , 1-1e-7) # to avoid 0 , log(0) == inf
    self.output = -np.log(encode)

    return self.output
