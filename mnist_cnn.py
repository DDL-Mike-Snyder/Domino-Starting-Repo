import numpy as np
import tensorflow as tf

MODEL_PATH = "mnist_cnn.keras"


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


def train_and_evaluate(epochs=3):
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, verbose=2)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    model.save(MODEL_PATH)
    return model, accuracy


def predict_digit(image, model=None):
    """Predict the digit in a handwritten image.

    image: array-like, shape (28, 28) or (28, 28, 1), values in [0, 255] or [0, 1].
    model: a loaded tf.keras.Model; if None, loads MODEL_PATH from disk.
    Returns (predicted_digit, confidence).
    """
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    image = np.asarray(image, dtype="float32")
    if image.max() > 1.0:
        image = image / 255.0
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)

    probabilities = model.predict(image, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])
    return predicted_digit, confidence


if __name__ == "__main__":
    trained_model, test_accuracy = train_and_evaluate()

    (_, _), (x_test, y_test) = load_data()
    sample_image, true_label = x_test[0], y_test[0]
    predicted_digit, confidence = predict_digit(sample_image, model=trained_model)
    print(f"Sample prediction: {predicted_digit} (confidence {confidence:.4f}), true label: {true_label}")
