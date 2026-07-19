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

Weights are initialized with **He initialization** (\(\sqrt{2/n_{in}}\)) for stable training with ReLU activations.

![Description](arcitecture_stuff/dense.jpg)
