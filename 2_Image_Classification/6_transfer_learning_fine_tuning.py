# -*- coding: utf-8 -*-
"""05_transfer_learning_fine_tuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14A-cbhQJE2WD-YeFyPx8frBIjgkXxbF0

# Transfer learning with fine tuning
"""

!nvidia-smi

# Get helper_functions.py script from course GitHub
!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py 

# Import helper functions we're going to use
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

"""## Let's get some data

Also, see how to use `tf.keras.applications` to use pre-train models
"""

# Get 10% of the data of the 10 classes
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip 

unzip_data("10_food_classes_10_percent.zip")

walk_through_dir("10_food_classes_10_percent")

# Create training and testing directory paths

train_dir = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"

import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_data_10_percent = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                    image_size=IMG_SIZE,
                                                                    label_mode="categorical",
                                                                    batch_size=BATCH_SIZE)
test_data = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                            image_size=IMG_SIZE,
                                                            label_mode="categorical",
                                                            batch_size=BATCH_SIZE)

train_data_10_percent

# check out the class names of our dataset
train_data_10_percent.class_names

# See an example of a batch of data
for batch in train_data_10_percent.take(1):
  images, labels = batch
  print(f"Image shape: {images[0].shape}, label shape:{labels[0].shape}")
  break

"""## Model 0: a transfer learning feature extraction model with functional API"""

# 1. Create a base model with tf.keras.applications
base_model = tf.keras.applications.EfficientNetB0(include_top=False)

# 2. Freeze the base model
base_model.trainable = False

# 3. Create inputs into our model
inputs = tf.keras.layers.Input(shape=(224,224,3), name="input_layer")

# 4. If using a model like ResNet50V2, will need to noramlize input, not applicable for ResNet
# x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

# 5. Pass the inputs to the base_model
x = base_model(inputs)
print(f"Shape after passing inputs through the base model: {x.shape}")

x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pooling_layer")(x)
print(f"Shape after global_avg_pooling_layer: {x.shape}")

outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layers")(x)

model_0 = tf.keras.models.Model(inputs=inputs,
                                outputs=outputs)

model_0.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics="accuracy")


history_0 = model_0.fit(train_data_10_percent,
            epochs=5,
            validation_data=test_data)

model_0.evaluate(test_data)

for layer_number, layer in enumerate(model_0.layers):
  print(layer_number, layer)

for layer_number, layer in enumerate(base_model.layers):
  print(layer_number, layer.name)

base_model.summary()

model_0.summary()

plot_loss_curves(history_0)

"""## Getting a feature vector from a pre-trained model"""

# Test the GlobalAveragePooling

input_shape = (1, 4, 4, 3)

tf.random.set_seed(42)

input_tensor = tf.random.normal(input_shape)
print(f"Random input tensor: {input_tensor}")

pooled_output = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
pooled_output.numpy()

# Let's replicate the global average 2D layer
tf.reduce_mean(input_tensor, axis=[1,2]).numpy()

"""## Add data augmentation right into the model

- with `tf.keras.layers.experimental.preprocessing()`, it support to run the preprocessing with a GPU.

- It only happens in training.

⚠**Note** the pre-processing layers is deprecated, can use something like tf.keras.layers.RandomFlip() layer instead.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

data_augmentation = keras.Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.2),
  preprocessing.RandomZoom(0.2),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2),
  # preprocessing.Rescaling(1./255)
], name="data_augmentation")

# Download and unzip data
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip
unzip_data("10_food_classes_1_percent.zip")

# Create training and test dirs
train_dir_1_percent = "10_food_classes_1_percent/train/"
test_dir = "10_food_classes_1_percent/test/"

IMG_SIZE = (224, 224)
train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_1_percent,
                                                                           label_mode="categorical",
                                                                           batch_size=32, # default
                                                                           image_size=IMG_SIZE)
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE)

# View a random image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random


target_class = random.choice(train_data_1_percent.class_names) # choose a random class
target_dir = "10_food_classes_1_percent/train/" + target_class # create the target directory
random_image = random.choice(os.listdir(target_dir)) # choose a random image from target directory
random_image_path = target_dir + "/" + random_image # create the choosen random image path
img = mpimg.imread(random_image_path) # read in the chosen target image
plt.imshow(img) # plot the target image
plt.title(f"Original random image from class: {target_class}")
plt.axis(False); # turn off the axes

# Augment the image
augmented_img = data_augmentation(tf.expand_dims(img, axis=0), training=True) # data augmentation model requires shape (None, height, width, 3)
plt.figure()
# plt.imshow(tf.squeeze(augmented_img)/255.) # requires normalization after augmentation
plt.imshow(tf.squeeze(augmented_img)) # requires normalization after augmentation

plt.title(f"Augmented random image from class: {target_class}")
plt.axis(False);

"""# Model 1: Feature extraction layer transfer learning"""

tf.random.set_seed(42)
tf.keras.backend.clear_session()

input_shape = (224,224,3)
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False


inputs = layers.Input(shape=input_shape, name="input_layer")

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)

model_1 = keras.models.Model(inputs, outputs)

model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

"""Model checkpoint callback"""

# Setup checkpoint path
checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt" # note: remember saving directly to Colab is temporary

# Create a ModelCheckpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, # set to False to save the entire model
                                                         save_best_only=False, # set to True to save only the best model instead of a model every epoch 
                                                         save_freq="epoch", # save every epoch
                                                         verbose=1)

history_10_percent = model_1.fit(train_data_10_percent,
                                epochs=5,
                                steps_per_epoch=len(train_data_10_percent),
                                validation_data=test_data,
                                validation_steps=int(0.25 * len(test_data)),
                                callbacks=[create_tensorboard_callback(dir_name="vision",
                                                                       experiment_name="1_percent_data_aug"),
                                           checkpoint_callback])

plot_loss_curves(history_10_percent)

model_1.evaluate(test_data)

model_1.summary()

data_augmentation.summary()

"""### Loading in checkpointed weights"""

model_1.load_weights(checkpoint_path)
loaded_weights_model_results = model_1.evaluate(test_data)

# the precision of digit storing leads to nunances, but mostly the results are the same with the prior evaluation result
loaded_weights_model_results

"""# Fine tuning transfer learning models"""

model_1.layers

for layer in model_1.layers:
  print(layer, layer.trainable)

# The efficient net layers are not trainable
for i, layer in enumerate(model_1.layers[2].layers):
  print(i, layer.name, layer.trainable)

print(f"Number of layers of base model: {len(base_model.layers)}")

# To begin fine-tuning, set the last 10 layers of the base model to be True
base_model.trainable = True

# Freeze all layers except for the last 10
for layer in base_model.layers[:-10]:
  layer.trainable = False

# Recompile the model once there is any change
model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(lr=1e-4), #reduce the lr when fine-tuning by 10x
                metrics=["accuracy"])

# Fine tune for another 5 epochs
initial_epochs = 5
fine_tune_epochs = initial_epochs + 5


history_fine_tune_data_aug = model_1.fit(train_data_10_percent,
              epochs=fine_tune_epochs,
              validation_data=test_data,
              validation_steps=int(0.25 * len(test_data)),
              initial_epoch=history_10_percent.epoch[-1],
              callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                    experiment_name="10_percent_fine_tune")])

model_1.evaluate(test_data)

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

import matplotlib.pyplot as plt
compare_historys(original_history=history_10_percent, 
                 new_history=history_fine_tune_data_aug, 
                 initial_epochs=5)