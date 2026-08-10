import numpy as np 


class Maxpooling:

  def __init__(self , pool_size = 2 , stride = 2):
    self.pool_size = pool_size
    self.stride = stride

  def forward(self , input):

    self.input = input
    batch ,channel , height , width = input.shape

    out_h = (height - self.pool_size) // self.stride + 1
    out_w = (width - self.pool_size) // self.stride + 1

    self.output = np.zeros((batch , channel , out_h ,out_w))
    self.max_index = np.zeros((batch , channel , out_h , out_w ,2))

    for b in range(batch):
      for c in range(channel):
        for h in range(0 , (height - self.pool_size) + 1 , self.stride):
          for w in range(0 , (width - self.pool_size) + 1 , self.stride):

            patch = input[b,c ,
                                h : h + self.pool_size ,
                                w : w + self.pool_size]

            self.output[b,c,h//self.stride, w//self.stride] = np.max(patch)

            max_pos = np.unravel_index(np.argmax(patch) ,patch.shape)

            self.max_index[b,c,h//self.stride ,w//self.stride] = (max_pos[0] ,max_pos[1])

    return self.output


  def backward(self , dl_dp):

    self.dl_di = np.zeros_like(self.input ,dtype=float)
    batch , channel  , height , width = self.input.shape
    mask = np.zeros((self.pool_size , self.pool_size))

    for b in range(batch):
      for c in range(channel):
        for h in range(0 ,(height - self.pool_size) + 1 ,self.stride):
          for w in range(0 ,( width - self.pool_size) + 1, self.stride):

            max_pos_x , max_pos_y = self.max_index[b,c,h//self.stride ,w//self.stride]
            
            mask[int(max_pos_x) ,int(max_pos_y)] = 1
            self.dl_di[
                        b, c,
                        h:h+self.pool_size,
                        w:w+self.pool_size
                 ] += mask * dl_dp[b, c, h//self.stride, w//self.stride]
                                
   return self.dl_di
  
