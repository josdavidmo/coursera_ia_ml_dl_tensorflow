from tensorflow import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.MeanSquaredError())

xs = np.array([1.0, 2.0, 3.0, 5.0], dtype=float)
ys = np.array([10836.0, 27386.0, 43285.0, 75256.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([4.0]))
