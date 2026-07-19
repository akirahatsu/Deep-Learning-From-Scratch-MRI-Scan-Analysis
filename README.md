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

<img src="Deep-Learning-From-Scratch-MRI-Scan-Analysis/img/dense.jpg" width="700">
<img src="Deep-Learning-From-Scratch-MRI-Scan-Analysis/img/dense.jpg" width="700">
