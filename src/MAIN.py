import import matplotlib.pyplot as plt
import numpy as plt


loss_history = []
accuracy_history = []

cnn1 = CNN(kernel_shape=(3,3), n_kernel=8)
activation1 = Relu()
mx1 = Maxpooling()

cnn2 = CNN(kernel_shape=(3,3), n_kernel=16)
activation2 = Relu()
mx2 = Maxpooling()

cnn3 = CNN(kernel_shape=(3,3), n_kernel=32)
activation3 = Relu()
mx3 = Maxpooling()

flatten = Flatten()
layer = Dense(5)

loss = Soft_Categoricalcrossentropy()

optimiser = SGD(decay=0.001, learning_rate=0.05)

batch_size = 32
epochs = 20

for epoch in range(epochs):

    # Shuffle dataset
    indices = np.random.permutation(len(images))

    images_shuffled = images[indices]
    labels_shuffled = labels[indices]

    epoch_loss = 0
    epoch_accuracy = 0
    num_batches = 0

    for start in range(0, len(images), batch_size):

        end = start + batch_size

        X_batch = images_shuffled[start:end]
        y_batch = labels_shuffled[start:end]

        # =====================
        # FORWARD
        # =====================

        cnn1.forward(X_batch)
        activation1.forward(cnn1.output)
        mx1.forward(activation1.output)

        cnn2.forward(mx1.output)
        activation2.forward(cnn2.output)
        mx2.forward(activation2.output)

        cnn3.forward(mx2.output)
        activation3.forward(cnn3.output)
        mx3.forward(activation3.output)

        flatten.forward(mx3.output)

        layer.forward(flatten.output)

        current_loss = loss.forward(
            layer.output,
            y_batch
        )

        current_accuracy = accuracy(
            loss.soft.output,
            y_batch
        )

        epoch_loss += current_loss
        epoch_accuracy += current_accuracy
        num_batches += 1

        # =====================
        # BACKWARD
        # =====================

        loss.backward(y_batch)

        layer.backward(loss.dl_dz)

        flatten.backward(layer.dl_di)

        mx3.backward(flatten.dl_dp)
        activation3.backward(mx3.dl_di)
        cnn3.backward(activation3.dl_dz)

        mx2.backward(cnn3.dl_di)
        activation2.backward(mx2.dl_di)
        cnn2.backward(activation2.dl_dz)

        mx1.backward(cnn2.dl_di)
        activation1.backward(mx1.dl_di)
        cnn1.backward(activation1.dl_dz)

        # =====================
        # UPDATE
        # =====================

        optimiser.itteration_func()
        optimiser.learning_rate_update()

        optimiser.update(cnn1)
        optimiser.update(cnn2)
        optimiser.update(cnn3)
        optimiser.update(layer)

    # =====================
    # EPOCH RESULT
    # =====================

    epoch_loss /= num_batches
    epoch_accuracy /= num_batches

    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)

    print(
        f"epoch: {epoch} "
        f"loss: {epoch_loss:.4f} "
        f"accuracy: {epoch_accuracy:.4f}"
    )

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
