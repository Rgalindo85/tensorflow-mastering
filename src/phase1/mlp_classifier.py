import logging
import tensorflow as tf

def main():
    logging.info("Starting MLP Classifier training")
    x = tf.random.normal((1000, 20))
    y = tf.random.uniform((1000,), maxval=2, dtype=tf.int32)

    classifier = MLPClassifier(input_dim=20, num_classes=2)
    classifier.train(x, y, epochs=10, batch_size=32)

    x_test = tf.random.normal((10, 20))
    predictions = classifier.predict(x_test)
    predicted_classes = classifier.predict_classes(x_test)
    print(f"Predictions: {predictions}")
    print(f"Predicted classes: {predicted_classes}")


class MLPClassifier:
    def __init__(self, input_dim: int = 20, num_classes: int = 2):
        # Define a simple MLP model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    def train(self, x, y, epochs: int=10, batch_size: int=32):
        # Compile and train the model
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy']
                           )
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
    
    def predict(self, x):
        return self.model.predict(x)
    
    def predict_classes(self, x):
        predictions = self.model.predict(x)
        return tf.argmax(predictions, axis=1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    main()