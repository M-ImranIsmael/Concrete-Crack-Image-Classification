#%% Libraries
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, losses, callbacks, applications

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

#%% Data Loading

data_path = os.path.join(os.getcwd(), "datasets")

#%%
# Define batch size and image size
BATCH_SIZE = 32
IMG_SIZE = (227, 227)

dataset = keras.utils.image_dataset_from_directory(
    data_path, batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True
)
class_names = dataset.class_names

#%% Splitting into train-val-test (80-10-10)

dataset_batches = tf.data.experimental.cardinality(dataset)
train_dataset = dataset.skip(dataset_batches // 5)
validation_dataset = dataset.take(dataset_batches // 5)

val_batches = tf.data.experimental.cardinality(validation_dataset)
validation_dataset = validation_dataset.skip(val_batches // 2)
test_dataset = validation_dataset.take(val_batches // 2)

#%% EDA

# Plot some examples
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# %% Convert the train-val-test into PrefetchDatasets
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_validation = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)


# %% Create a model for image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip("horizontal"))
data_augmentation.add(layers.RandomRotation(0.2))

#%% Test out the data augmentation layers
for images, labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, axis=0))
        plt.imshow(augmented_image[0] / 255.0)
        plt.axis("off")
plt.show()

# %% Transfer Learning
"""
In this example, we are applying transfer learning on MobileNetV2 that is pretrained with imagenet dataset
"""

# Create a layer to preprocess input
preprocess_input = applications.mobilenet_v2.preprocess_input

#%% A) Instantiate the pretrained model

IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
)

#%% B) Set the pretrained feature extractor as non-trainable (same as freezing the layer)

base_model.trainable = False
base_model.summary()
keras.utils.plot_model(base_model, show_shapes=True)

# %% C) Create the classification layer
# Create global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
# Create an output layer
output_layer = layers.Dense(len(class_names), activation="softmax")

# Link the model pipeline using functional
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

# %% Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# %% Evaluate with model before training
loss0, acc0 = model.evaluate(pf_test)
print("================Evaluation Before Training================")
print(f"Loss = {loss0}")
print(f"Accuracy = {acc0}")

# %% TensorBoard callbacks and model training
base_log_path = r"tensorboard_logs"
log_path = os.path.join(
    base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
tb = callbacks.TensorBoard(log_path)

EPOCHS = 3
history = model.fit(
    pf_train, validation_data=pf_validation, epochs=EPOCHS, callbacks=[tb]
)

# %% Test with test data
test_loss, test_acc = model.evaluate(pf_test)
print("================Evaluation After Training================")
print(f"Loss = {test_loss}")
print(f"Accuracy = {test_acc}")

# %% Model Deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch), axis=1)

# Stack the label and prediction in one numpy array
label_vs_prediction = np.transpose(np.vstack((label_batch, y_pred)))

# Save the model
save_path = os.path.join("save_model", "concrete_cracks_model.h5")
model.save(save_path)


# %% Load the model for testing
loaded_model = keras.models.load_model(save_path)
loaded_model.summary()

#%% Generate predictions on test dataset
predictions = loaded_model.predict(pf_test)

# Convert predictions from one-hot encoding to class indices
predicted_labels = np.argmax(predictions, axis=1)

# Plot test data with preicted labels
for images, labels in pf_test.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Predicted {class_names[predicted_labels[i]]}")
        plt.axis("off")
plt.show()

#%% Generating confusion matrix

true_labels = []
predicted_labels = []

for images, labels in pf_test:
    predictions = loaded_model.predict(images)
    predicted_labels_batch = np.argmax(predictions, axis=1)
    true_labels.extend(labels)
    predicted_labels.extend(predicted_labels_batch)

cm = confusion_matrix(true_labels, predicted_labels)
print(cm)

#%% Display confusion matrix
labels = ["Positive", "Negative"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)

#%% Generating classification report
cr = classification_report(true_labels, predicted_labels, target_names=class_names)
print(cr)

#%%
# Plot the images with their true and predicted labels
plt.figure(figsize=(10, 10))
for i in range(len(image_batch)):
    for i in range(9):
        plt.subplot(4, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(
            f"True label: {class_names[label_batch[i]]}\nPredicted label: {class_names[y_pred[i]]}"
        )
        plt.axis("off")
plt.show()
