import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
  """
  Custom Callback in order to stop it when the loss is low.
  """

  def on_epoch_end(self, epoch, logs=None):
    if (logs.get('loss') < 0.4):
      print("\n Loss is low so cancelling training!")
      self.model.stop_training = True


# Get the data and split it into training and test data
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (
  test_images, test_labels) = mnist.load_data()
plt.imshow(training_images[0])
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
  tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(training_images,
          training_labels,
          epochs=5,
          workers=8,
          use_multiprocessing=True,
          callbacks=[myCallback()])
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

# Each probability for labels
print(classifications[0])
