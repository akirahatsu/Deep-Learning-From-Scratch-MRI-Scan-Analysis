import numpy as np

class Momentum:

  def __init__(self ,learning_rate ,decay = 0.005 , momentum = 0):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.itteration = 0
    self.momentum = momentum

  def pre_update(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * \
       (1/(1+self.decay * self.itteration))


  def optimise(self , layer_optim):
    if self.momentum:
      # Initialize momentums if they don't exist for this layer
      if not hasattr(layer_optim,'weight_momentums'):
        layer_optim.weight_momentums = np.zeros_like(layer_optim.weight)
        layer_optim.bias_momentums = np.zeros_like(layer_optim.bias)

      # Calculate momentum updates
      weight_updates = (self.momentum * layer_optim.weight_momentums) - (self.current_learning_rate * layer_optim.dl_dw)
      bias_updates = (self.momentum * layer_optim.bias_momentums) - (self.current_learning_rate * layer_optim.dl_db)

      # Store the updates for the next iteration (this is the new momentum)
      layer_optim.weight_momentums = weight_updates
      layer_optim.bias_momentums = bias_updates

    else:
      # Standard SGD update without momentum
      weight_updates = -self.current_learning_rate * layer_optim.dl_dw
      bias_updates = -self.current_learning_rate * layer_optim.dl_db

    # Apply updates to weights and biases
    layer_optim.weight += weight_updates
    layer_optim.bias += bias_updates

  def itteration_update(self):
    self.itteration += 1


class SGD:

  def __init__(self ,learning_rate ,decay = 0.005):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.itteration = 0

  def pre_update(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * \
       (1/(1+self.decay * self.itteration))


  def optimise(self , layer_optim):

    layer_optim.weight -= self.current_learning_rate * layer_optim.dl_dw
    layer_optim.bias -= self.current_learning_rate * layer_optim.dl_db

  def itteration_update(self):
    self.itteration += 1   


class VanillaGD:

  def __init__(self ,learning_rate):
    self.learning_rate = learning_rate

  def optimise(self , layer_optim):

    layer_optim.weight -= self.learning_rate * layer_optim.dl_dw
    layer_optim.bias -= self.learning_rate * layer_optim.dl_db



class ADAM:
  def __init__(self , decay = 0.005 , learning_rate = 0.01 , rho = 0.99 , epsilon= 1e-7  , beta = 0.9):
    self.decay = decay
    self.learning_rate = learning_rate
    self.current_learning_rate = 0
    self.itteration = 0
    self.rho = rho
    self.beta = beta
    self.epsilon = epsilon


  def pre_update(self):
    self.current_learning_rate = self.learning_rate * (1/(1+ self.decay * self.itteration))

  def optimise(self , layer_optim):

    # RMSprop

    if not hasattr(layer_optim , 'weight_cache'):
      #Cache w/b
      layer_optim.weight_cache = np.zeros_like(layer_optim.weight)
      layer_optim.bias_cache =  np.zeros_like(layer_optim.bias)
      #Momentum w/ b
      layer_optim.weight_momentum = np.zeros_like(layer_optim.weight)
      layer_optim.bias_momentum = np.zeros_like(layer_optim.bias)


    weight_cache_update = self.rho * layer_optim.weight_cache + (1-self.rho) * layer_optim.dl_dw ** 2
    bias_cache_update = self.rho * layer_optim.bias_cache + (1-self.rho) * layer_optim.dl_db ** 2

    layer_optim.weight_cache = weight_cache_update
    layer_optim.bias_cache = bias_cache_update


    # Momentum

    weight_momentum_update = self.beta *  layer_optim.weight_momentum + (1 - self.beta) * layer_optim.dl_dw
    bias_momentum_update = self.beta * layer_optim.bias_momentum + (1-self.beta) * layer_optim.dl_db

    layer_optim.weight_momentum = weight_momentum_update
    layer_optim.bias_momentum = bias_momentum_update

    #Bias correction
    weight_cache_corrected = layer_optim.weight_cache / (1-self.rho ** (self.itteration + 1))
    bias_cache_corrected = layer_optim.bias_cache / ( 1- self.rho ** (self.itteration + 1))

    weight_momentum_corrected = layer_optim.weight_momentum / (1-self.beta ** (self.itteration + 1))
    bias_momentum_corrected = layer_optim.bias_momentum / (1-self.beta ** (self.itteration + 1))

    #Update

    layer_optim.weight -= self.current_learning_rate * weight_momentum_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
    layer_optim.bias -= self.current_learning_rate * bias_momentum_corrected / (np.sqrt(bias_cache_corrected)+self.epsilon)

  def itteration_update(self):
    self.itteration += 1


