import numpy as np
from activation import Softmax
from CategoricalCrossEntropyLoss import CatergoricalCrossEntropy



"""
Combining Softmax and Cross-Entropy allows us to compute the gradient
directly without explicitly constructing the Softmax Jacobian matrix.

The resulting gradient with respect to the logits is

    dL/dz = (softmax_output - ground_truth) / N

where N is the batch size when using the mean loss.
"""


class Softmax_Loss_crossentropy():
  

  # We initialise softmax and cross entropy tha we made before 

  def __init__(self):
    self.softmax = Softmax()
    self.crossentropyloss = CatergoricalCrossEntropy()

  # Forward
  def forward(self , input , y_true): 

    self.softmax.forward(input)
    self.output = self.crossentropyloss.forward(self.softmax.output ,y_true)
    self.y_true = y_true

    return self.output
  #Backword

  def backward(self):

    sample = self.softmax.output.shape[0]

    soft_output = self.softmax.output.copy() 

    soft_output[range(sample),self.y_true] -= 1 # softmax - groud_true
    
    # Loss with respect last layer output 
    self.dl_dz = soft_output / sample # mean derivative == 1/N

    return self.dl_dz
