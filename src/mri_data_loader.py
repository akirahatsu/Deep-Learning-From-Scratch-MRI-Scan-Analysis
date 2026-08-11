import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)

#seperating to classes 
classes = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

# load from path 
import os
from PIL import Image
import numpy as np
images = []
labels = []


TARGET_SIZE = (128, 128)

train_path = os.path.join(path, "Training")

for class_name, label in classes.items():
    folder = os.path.join(train_path, class_name)

    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)

        image = Image.open(image_path).convert("L")

        # Resize the image to the TARGET_SIZE

        image = image.resize(TARGET_SIZE)
        image = np.array(image)

        images.append(image)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

print(images.shape)
print(labels.shape)
images[0]

#permuataion
permutation = np.random.permutation(len(images))

images = images[:,np.newaxis ,: ,:]

images = images[permutation]
labels = labels[permutation]

#normoliseing 
labels = np.array(labels , dtype = int)
images = images / 255

