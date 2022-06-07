# Use pre-trained layers and ImageDataGenerator etc.
# =============================================
# API doc from TensorFlow Org: https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU
# Reference of convolutional layer, pooling layer can be found at《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》- Chapter 14
# Note: some APIs are deprecated so the reference in the book are tweaked in Colab and now it becomes like below


# Dependencies
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

dataset, info = tfds.load('tf_flowers', as_supervises=True, with_info=True)
dataset_size = info.split['train'].num_examples
class_names = info.features['labels'].names
n_classes = info.features['labels'].num_classes

# create test/valid/train sets, DEPRECATED!!!
# test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])

# New API doc: https://www.tensorflow.org/datasets/splits
test_set = tfds.load('tf_flowers', split='train[:75%]', as_supervised=True)
valid_set = tfds.load('tf_flowers', split='train[75%:90%]', as_supervised=True)
train_set = tfds.load('tf_flowers', split='train[90%:]', as_supervised=True)


# preprocess images
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    #     xception model also has a preprocess_input function
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


# Apply the preprocessing function to all images
batch_size = 32
epochs = 5
train_set = train_set.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size)
valid_set = valid_set.map(preprocess).batch(batch_size)
test_set = test_set.map(preprocess).batch(batch_size)

# keras.preprocessing.image.ImageDataGenerator is also a great API for load images and augment them, however it is to be deprecated somehow


# Load the pre-trained Xception model
base_model = keras.applications.xception.Xception(weights='imagenet', include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

# as a best practice, normally it is better to freeze the base model's weights
for layer in base_model.layers:
    layer.trainable = False

optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(train_set, epochs=epochs, validation_data=valid_set)


# visualize model results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
