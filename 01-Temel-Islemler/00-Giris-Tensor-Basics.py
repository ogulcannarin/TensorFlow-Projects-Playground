import tensorflow as tf 
import numpy as np

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile
model.compile(optimizer='sgd', loss='mean_squared_error')

# Data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train
model.fit(xs, ys, epochs=500)

# Prediction
print(model.predict(np.array([10.0])))