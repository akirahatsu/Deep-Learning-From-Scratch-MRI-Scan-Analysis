import import matplotlib.pyplot as plt
import numpy as plt


loss_history = []
accuracy_history = []

cnn1 = CNN(kernel_shape=(3,3) , n_kernel=8)
activation1 = Relu()
mx1 = Maxpooling()
cnn2 = CNN(kernel_shape=(3,3) , n_kernel=16)
activation2 = Relu()
mx2 = Maxpooling()
cnn3 = CNN(kernel_shape=(3,3) , n_kernel=32)
activation3 = Relu()
mx3 = Maxpooling()
flatten = Flatten()
layer = Dense(5)
loss =  Soft_Categoricalcrossentropy()

optimiser = SGD(decay=0.001 ,learning_rate=0.05)

for el in range(10):

  cnn1.forward(images[:4])
  activation1.forward(cnn1.output)
  mx1.forward(activation1.output)

  # cnn2.forward(mx1.output)
  # activation2.forward(cnn2.output)
  # mx2.forward(activation2.output)

  # cnn3.forward(mx2.output)
  # activation3.forward(cnn3.output)
  # mx3.forward(activation3.output)

  flatten.forward(mx1.output)
  layer.forward(flatten.output)
  loss.forward(layer.output , labels[:4])

  # accuracy_rate = np.mean(np.argmax(loss.loss.soft - ))
  print(f"\nepoch : { el}")
  print(f"loss : {loss.loss.output_loss}")
  
  print(f"acuracy : {accuracy(loss.soft.output , labels[:4])}")
  loss_history.append(loss.loss.output_loss)
  accuracy_history.append(accuracy(loss.soft.output , labels[:4]))

  # print("\n\nsoftmax output:")
  # print(loss.soft.output)
  # print("\n\nfinal logits:")
  # print(layer.output)

  # print("std of logits:")
  # print(np.std(layer.output))


  loss.backward(labels[:4])
  layer.backward(loss.dl_dz)
  flatten.backward(layer.dl_di)
  mx1.backward(flatten.dl_dp)
  activation1.backward(mx1.dl_di)
  cnn1.backward(activation1.dl_dz)

  optimiser.itteration_func()
  optimiser.learning_rate_update()
  optimiser.update(cnn1)
  optimiser.update(layer)



plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

plt.plot(accuracy_history)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()
