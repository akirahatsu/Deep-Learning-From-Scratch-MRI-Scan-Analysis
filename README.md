# Deep-Learning-From-Scratch-MRI-Scan-Analysis
Research-oriented implementation of neural networks from first principles, demonstrating mathematical foundations and medical image classification.

## Dense Layer

Implements a fully connected (dense) layer using matrix multiplication:

Z = XW + b

The forward pass computes the affine transformation, while the backward pass derives gradients with respect to the weights, biases, and inputs using the multivariable chain rule:

Backward:

dL/dW = Xᵀδ

dL/db = Σδ

dL/dX = δWᵀ


Weights are initialized with **He initialization** np.sqrt(2 / n_input) for stable training with ReLU activations.

<img src="dense.jpg" width="700">

CATEGORICAL CROSS ENTROPY

if ndim == 2 --> one hot encoded 

SAMPLE 

one_hot_encoded = np.array(

               [[1,0,0],

               [0,1,0],
               
               [1,0,0],
               
               [0,0,1]])

               
input = np.array(

                 [[ 0., 0., 0.],

                 [-0.00994372, -0.00501981, -0.00084307],
                 
                 [ 0.01182119, -0.00391712, -0.00248948],
                 
                 [-0.00968822, -0.01280488, -0.00361701]]) 
                 

np.sum( 0one_hot_encoded * input , axis = 1 , keepdims = 1)


output = np.array(

                  [[ 0.],

                  [-0.00501981],
                  
                  [ 0.01182119],
                  
                  [-0.00361701]])
                  
/////////

elif ndim == 1 list format

l_encode_target = np.array(

                 [0,1,0,2]  )    

input[ range ( len( l_encode_target) ), l_encode_target ].reshape(-1,1) # reshape part broudcasting 


output = np.array(

                  [[ 0.],
                 
                  [-0.00501981],
                 
                  [ 0.01182119],
                
                  [-0.00361701]])


self.output = -np.log(output)   # -log and use np.clip prevent overflow          
               
