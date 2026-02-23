import numpy as np
import tensorflow as tf
# GRADED FUNCTION: create_training_data

def create_training_data():
    """Creates the data that will be used for training the model.

    Returns:
        (numpy.ndarray, numpy.ndarray): Arrays that contain info about the number of bedrooms and price in hundreds of thousands for 6 houses.
    """
    
    # Features (number of bedrooms)
    n_bedrooms = np.array([1., 2., 3., 4., 5., 6.], dtype=float)

    # Targets (price in hundreds of thousands)
    price_in_hundreds_of_thousands = np.array([1., 1.5, 2., 2.5, 3., 3.5], dtype=float)

    return n_bedrooms, price_in_hundreds_of_thousands


def define_and_compile_model():
    """Returns the compiled (but untrained) model."""

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer='sgd',
        loss='mean_squared_error'
    )

    return model

# GRADED FUNCTION: train_model

def train_model():
    """Returns the trained model."""

    # Get training data
    n_bedrooms, price_in_hundreds_of_thousands = create_training_data()

    # Get compiled model
    model = define_and_compile_model()

    # Train model
    model.fit(n_bedrooms, price_in_hundreds_of_thousands, epochs=500)

    return model

trained_model = train_model()

new_n_bedrooms = np.array([7.0])
predicted_price = trained_model.predict(new_n_bedrooms, verbose=False).item()

print(f"Prediction: {predicted_price:.2f}")