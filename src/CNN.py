class CNN():

  def __init__(self, kernel_shape):

    self.kernel = np.random.randn(*kernel_shape)
    self.bias = np.zeros((kernel_shape[0]))

  def forward(self,input):


    self.input = np.array(input)

    self.signal_shape = np.array(self.input).shape

    self.output = np.zeros(
       (self.kernel.shape[0] ,
       self.signal_shape[0]-self.kernel.shape[1] + 1,
      self.signal_shape[1]-self.kernel.shape[2] + 1))

    for k in range(self.kernel.shape[0]):



        for el in range(self.signal_shape[0]-self.kernel.shape[1] + 1):
            for i in range(self.signal_shape[1]-self.kernel.shape[2] + 1):

                patch = self.input[
                     el : el+self.kernel.shape[1] ,
                     i :i+self.kernel.shape[2]
                     ]

                self.output[k ,el,i] = np.sum(patch * self.kernel[k]) + self.bias[k]

    return self.output

  def backward(self , dl_dp):

    self.dl_dw = np.zeros_like(self.kernel)
    self.dl_db = np.zeros((self.kernel.shape[0]))
    self.dl_di = np.zeros_like(self.input, dtype=float) 

    for k in range(self.kernel.shape[0]):

      for el in range(self.signal_shape[0] - self.kernel.shape[1] + 1):

        for i in range(self.signal_shape[1] - self.kernel.shape[2] + 1) :

          patch = self.input[
            el : el + self.kernel.shape[1] ,
            i :i + self.kernel.shape[2]
            ]

          self.dl_dw[k] += patch * dl_dp[k,el,i]
          self.dl_db[k] += dl_dp[k,el,i]

          self.dl_di[
            el : el + self.kernel.shape[1] ,
            i :i + self.kernel.shape[2]
            ] += self.kernel[k] * dl_dp[k,el,i]

    return  self.dl_di
