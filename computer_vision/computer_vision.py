from tensorflow import keras
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=((28, 28))),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])
