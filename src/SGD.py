
import numpy as np

class SGD:

  def __init__(self , decay = 0  , learning_rate = 0.05):
    self.decay = decay
    self.learning_rate = learning_rate
    self.itteration = 0


  def learning_rate_update(self):
      self.learning_rate = self.learning_rate * (1/(1 + self.decay * self.itteration))

  def update(self,layer):
    self.layer.weight -= self.learning_rate * layer.weight
    self.layer.bias -= self.learning_rate * layer.bias

  def itteration_func(self):
    self.itteration += 1
