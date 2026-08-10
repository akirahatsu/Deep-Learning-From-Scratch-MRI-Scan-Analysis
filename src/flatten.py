import numpy as np

class Flatten:

  def forward(self,input):

    self.input = input
    batch = self.input.shape[0]

    self.output = input.reshape(batch , -1)

    return self.output

  def backward(self, dl_df):

    dl_dp = dl_df.reshape(self.input.shape)

    return dl_dp  

