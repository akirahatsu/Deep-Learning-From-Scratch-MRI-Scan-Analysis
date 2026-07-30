import numpy as np

class MaxPool():

  def __init__(self , pool_size = (2,2) ,stride  = 2) :

    self.pool_h = pool_size[0]
    self.pool_w  =  pool_size[1]
    self.stride = stride

  def forward(self,input):  

    batch , channel , height , width = input.shape

    output_h = (height - self.pool_h) // stride + 1
    output_w = (width - self.pool_w) // stride + 1

    self.output = np.zeros((batch , channel , output_h , output_w))
    self.max_index =  np.zeros((batch , channel , output_h ,output_w , 2) , dtype=int)


    for b in range(batch):
      for c in range(channel):
        for h in range(0 , (height - pool_h) + 1 , stride) :
          for w in range(0 , width - pool_w + 1 , stride):

            patch = input[b,c,h :h +pool_h , w : w + pool_w]

            self.output[b , c , h//stride , w//stride] = np.max(patch)
            
            max_pos = np.unravel_index(np.argmax(patch) , patch.shape)

            self.max_index[b,c,h//stride , w//stride] = (h + max_pos[0] , w + max_pos[1])


    return self.output   , self.max_index  
