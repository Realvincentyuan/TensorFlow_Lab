# Understand and use below components:
# - tf.keras.preprocessing.image.ImageDataGenerator
# - tf.keras.preprocessing.image_dataset_from_directory
# - tf.data.Dataset with image files
# - tf.data.Dataset with TFRecords
# =============================================
# API doc from TensorFlow Org: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_dataframe
import os
import tensorflow as tf
from glob import glob
import random
import math


def build_model(num_classes, img_size=224):
    input = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    model = tf.keras.applications.EfficientNetB3(include_top=False, input_tensor=input, weights='imagenet')

    # Freeze the pretained weights
    model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    x = tf.keras.layers.BatchNormalization(x)

    top_drop_out_rate = 0.2
    x = tf.keras.layers.Dropout(top_drop_out_rate, name='top_dropout')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='pred')(x)

    # Compile
    model = tf.keras.Model(inputs=input, outputs=output, name='EfficientNet')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["sparse_categorical_accuracy"])
    return model


# ImageDataGenerator
def use_keras_generators(path, img_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    dataset = datagen.flow_from_directory(path, (img_size, img_size), batch_size=32, class_mode='sparse')

    num_classes = len(os.listdir(path))
    model = build_model(num_classes)

    model.fit(dataset, batch_size=32, epochs=5)


# Image_dataset_from_directory
def use_keras_idfd(path):
    keras_ds = tf.keras.preprocessing.image_dataset_from_directory(path, batch_size=32, image_size=(img_size, img_size))
    keras_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    num_classes = len(os.listdir(path))
    model = build_model(num_classes)

    model.fit(keras_ds, batch_size=32, epochs=5)


# tf.data.Dataset
def make_dataset(path, batch_size, img_size):
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        return image

    def configure_for_performance(ds):
        ds = ds.shuffle(buffer_size=1000)
        df = ds.batch(batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    classes = os.listdir(path)
    file_names = glob(path + '/*/*')
    random.shuffle(file_names)
    labels = [classes.inex(name.split('/'[-2])) for name in file_names]

    filename_ds = tf.data.Dataset.from_tensor_slices(file_names)
    image_ds = filename_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((image_ds, labels_ds))
    ds = configure_for_performance(ds)

    return ds


def use_tf_data(path, img_size):
    dataset = make_dataset(path, 32, img_size)
    num_classes = os.listdir(path)
    num_images = len(glob(path + '/*/*'))
    model = build_model(num_classes, img_size)

    model.fit(dataset, batch_size=32, epochs=5, steps_per_epoch=math.ceil(num_images / 32))


# Image augmentation on the Tensorflow Dataset can be found at: https://www.tensorflow.org/tutorials/images/data_augmentation


# Writing TF records
def serialize_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def make_tfrecords(path, record_file='/content/images.tfrecords'):
    classes = os.listdir(path)
    with tf.io.TFRecordWriter(record_file) as writer:
        file_list = glob(path + '/*/*')
        random.shuffle(file_list)
        for filename in file_list:
            image_string = open(filename, 'rb').read()
            category = filename.split('/')[-2]
            label = classes.index(category)
            tf_example = serialize_example(image_string, label)
            writer.write(tf_example)


# Reading from TF records
def _parse_image_function(example, img_size):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_image(features['image'], channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    label = tf.cast(features['label'], tf.int32)
    return image, label


def read_dataset(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(500)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def use_tfrecords(path, img_size):
    dataset = read_dataset('/content/image.tfrecords',32)

    num_classes = len(os.listdir(path))
    num_images = len(glob(path +'*/*'))
    model = build_model(num_classes, img_size)

    model.fit(dataset, batch_size=32, epochs=5, steps_per_epoch=math.ceil(num_images/32))

