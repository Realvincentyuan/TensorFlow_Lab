# Convolutional layer
# =============================================
# Sample code from 《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》- Chapter 14

# Dependencies
from sklearn.datasets import load_sample_image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

china = load_sample_image('china.jpg') / 255
flower = load_sample_image('flower.jpg') / 255
images = np.array([china, flower])

batch_size, height, width, channels = images.shape

print(batch_size, height, width, channels)

# create 2 filters
filters = np.zeros(shape=(5, 5, channels, 2), dtype=np.float32)
filters[:, 2, :, 0] = 2  # middle vertical line set to 1
filters[2, :, :, 1] = 2  # middle horizontal line set to 1

output = tf.nn.conv2d(input=images, filters=filters, strides=1, padding='SAME')


# can show output of each filter
plt.imshow(output[1, :, :, 0], cmap='gray')
plt.show()

plt.imshow(output[1, :, :, 1], cmap='gray')

# show feature map
plt.show()
